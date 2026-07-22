/**
 * tokenizer.js — Client-side BPE encode/decode
 *
 * Exposes window.BPE = { init, encode, decode, loaded }
 */

window.BPE = (function () {
  let vocab = null;
  let merges = [];
  let mergeRank = new Map();
  let idToToken = new Map();
  let loaded = false;

  /**
   * Load tokenizer.json and initialise vocab + merges with retry logic.
   * @param {string} url - URL to fetch tokenizer from
   * @param {Object} options - Configuration options
   * @returns {Promise<void>}
   */
  async function init(url = 'data/tokenizer.json', options = {}) {
    const {
      maxRetries = 3,
      timeoutMs = 10000,
      retryDelay = 1000
    } = options;

    let lastError = null;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Loading tokenizer (attempt ${attempt}/${maxRetries})...`);
        
        // Create timeout controller
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
        
        try {
          const response = await fetch(url, { 
            signal: controller.signal,
            cache: 'no-cache' // Prevent aggressive caching issues
          });
          clearTimeout(timeoutId);
          
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
          
          const data = await response.json();
          
          // Validate required fields
          if (!data.vocab || !data.merges) {
            throw new Error('Invalid tokenizer format: missing vocab or merges');
          }
          
          vocab = data.vocab;
          merges = data.merges;
          
          // Build reverse lookup for decoding
          for (const [token, id] of Object.entries(vocab)) {
            idToToken.set(id, token);
          }
          
          // Build merge rank lookup for O(1) encoding
          for (let i = 0; i < merges.length; i++) {
            const [left, right] = merges[i];
            mergeRank.set(`${left},${right}`, i);
          }
          
          loaded = true;
          console.log(`Tokenizer loaded successfully on attempt ${attempt}`);
          return; // Success!
          
        } catch (fetchError) {
          clearTimeout(timeoutId);
          throw fetchError;
        }
        
      } catch (error) {
        lastError = error;
        
        // Determine if we should retry
        const isRetryable = 
          error.name === 'AbortError' || // Timeout
          error.message.includes('NetworkError') ||
          error.message.includes('Failed to fetch') ||
          (error.message.includes('HTTP') && 
           (error.message.includes('502') || error.message.includes('503') || error.message.includes('504')));
        
        if (attempt === maxRetries || !isRetryable) {
          break; // Don't retry on final attempt or non-retryable errors
        }
        
        console.warn(`Tokenizer load attempt ${attempt} failed:`, error.message);
        
        // Exponential backoff with jitter
        const delay = retryDelay * Math.pow(2, attempt - 1) + Math.random() * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    // If we get here, all retries failed
    console.error("Failed to load tokenizer after all retries:", lastError);
    
    // Enhanced error message with troubleshooting hints
    let errorMessage = `Failed to load tokenizer after ${maxRetries} attempts`;
    if (lastError?.name === 'AbortError') {
      errorMessage += ' (timeout - check your internet connection)';
    } else if (lastError?.message?.includes('HTTP')) {
      errorMessage += ` (${lastError.message} - server error)`;
    } else if (lastError?.message?.includes('NetworkError') || lastError?.message?.includes('Failed to fetch')) {
      errorMessage += ' (network error - check your connection or try refreshing)';
    } else {
      errorMessage += ` (${lastError?.message || 'unknown error'})`;
    }
    
    throw new Error(errorMessage);
  }

  /**
   * Greedy BPE encode (legacy character-level split).
   * @param {string} text
   * @returns {number[]} list of token IDs
   */
  function encode(text) {
    if (!loaded) {
      throw new Error("Tokenizer not loaded. Call init() first.");
    }
    
    // 1. Split text into initial token sequence = list of individual characters
    // Using Array.from to correctly handle surrogate pairs (emojis, etc.)
    let tokens = Array.from(text);
    
    // 2. While len(sequence) >= 2:
    while (tokens.length >= 2) {
      let bestRank = Infinity;
      let bestPair = null;
      
      // a. Compute all adjacent pairs and find the one with lowest rank
      for (let i = 0; i < tokens.length - 1; i++) {
        const pairKey = `${tokens[i]},${tokens[i+1]}`;
        const rank = mergeRank.has(pairKey) ? mergeRank.get(pairKey) : Infinity;
        
        if (rank < bestRank) {
          bestRank = rank;
          bestPair = [tokens[i], tokens[i+1]];
        }
      }
      
      // d. If no pair has finite rank: STOP
      if (bestRank === Infinity) {
        break;
      }
      
      // e. Merge all non-overlapping occurrences of that pair (left-to-right scan)
      const newTokens = [];
      let i = 0;
      while (i < tokens.length) {
        if (i < tokens.length - 1 && tokens[i] === bestPair[0] && tokens[i+1] === bestPair[1]) {
          newTokens.push(bestPair[0] + bestPair[1]);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i += 1;
        }
      }
      tokens = newTokens;
    }
    
    // 3. Return sequence of token IDs (via vocab lookup)
    const ids = [];
    for (const t of tokens) {
      if (t in vocab) {
        ids.push(vocab[t]);
      } else {
        // Unknown token fallback: encode each character individually
        for (const ch of Array.from(t)) {
          if (ch in vocab) {
            ids.push(vocab[ch]);
          }
          // silently skip characters truly not in vocab
        }
      }
    }
    
    return ids;
  }

  /**
   * BPE decode.
   * @param {number[]} ids
   * @returns {string}
   */
  function decode(ids) {
    if (!loaded) {
      throw new Error("Tokenizer not loaded. Call init() first.");
    }
    
    let result = '';
    for (const id of ids) {
      result += idToToken.has(id) ? idToToken.get(id) : "<UNK>";
    }
    return result;
  }

  return { init, encode, decode, get loaded() { return loaded; } };
})();
