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
   * Load tokenizer.json and initialise vocab + merges.
   * @returns {Promise<void>}
   */
  async function init(url = 'data/tokenizer.json') {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
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
    } catch (error) {
      console.error("Failed to load tokenizer:", error);
      throw error;
    }
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
