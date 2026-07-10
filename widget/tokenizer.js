/**
 * tokenizer.js — Client-side BPE encode/decode
 *
 * Exposes window.BPE = { init, encode, decode, loaded }
 *
 * Full implementation in Phase 5 (T023, T028).
 */

window.BPE = (function () {
  let vocab = null;
  let merges = null;
  let loaded = false;

  /**
   * Load tokenizer.json and initialise vocab + merges.
   * @returns {Promise<void>}
   */
  async function init(url = './data/tokenizer.json') {
    // TODO: T023 — implement fetch, parse vocab/merges, set loaded = true
  }

  /**
   * Greedy BPE encode.
   * @param {string} text
   * @returns {number[]} list of token IDs
   */
  function encode(text) {
    // TODO: T023 — port Python encode algorithm from contracts/tokenizer-format.md
    return [];
  }

  /**
   * BPE decode.
   * @param {number[]} ids
   * @returns {string}
   */
  function decode(ids) {
    // TODO: T023 — concatenate id_to_token lookups
    return '';
  }

  return { init, encode, decode, get loaded() { return loaded; } };
})();
