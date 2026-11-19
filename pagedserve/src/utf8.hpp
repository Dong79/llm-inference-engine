// Small UTF-8 helpers used by the tokenizer and by streaming detokenization.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ps::utf8 {

inline std::string encode(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

// Expected sequence length from a lead byte, or 0 if `b` cannot start one.
inline int seq_len(unsigned char b) {
  if (b < 0x80) return 1;
  if ((b & 0xE0) == 0xC0) return b >= 0xC2 ? 2 : 0;
  if ((b & 0xF0) == 0xE0) return 3;
  if ((b & 0xF8) == 0xF0) return b <= 0xF4 ? 4 : 0;
  return 0;
}

// Decodes one code point at s[i]; advances i. Returns 0xFFFD on malformed input.
inline uint32_t next(std::string_view s, size_t& i) {
  const unsigned char b0 = static_cast<unsigned char>(s[i]);
  const int n = seq_len(b0);
  if (n == 0 || i + n > s.size()) {
    ++i;
    return 0xFFFD;
  }
  uint32_t cp = n == 1 ? b0 : (b0 & (0x7F >> n));
  for (int k = 1; k < n; ++k) {
    const unsigned char c = static_cast<unsigned char>(s[i + k]);
    if ((c & 0xC0) != 0x80) {
      ++i;
      return 0xFFFD;
    }
    cp = (cp << 6) | (c & 0x3F);
  }
  i += n;
  return cp;
}

// Lossy conversion: malformed bytes become U+FFFD (like Rust from_utf8_lossy).
inline std::string sanitize(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const size_t start = i;
    const uint32_t cp = next(s, i);
    if (cp == 0xFFFD && !(i - start == 3 && s.substr(start, 3) == "\xEF\xBF\xBD"))
      out += "\xEF\xBF\xBD";
    else
      out.append(s.substr(start, i - start));
  }
  return out;
}

// For streaming: length of the longest prefix of `s` that does not end in an
// incomplete (but possibly still valid) multi-byte sequence. Bytes after it
// should be held back until the next token arrives.
inline size_t complete_prefix_len(std::string_view s) {
  const size_t n = s.size();
  for (size_t back = 1; back <= 4 && back <= n; ++back) {
    const unsigned char b = static_cast<unsigned char>(s[n - back]);
    if ((b & 0xC0) == 0x80) continue;  // continuation byte, keep looking for the lead
    const int need = seq_len(b);
    if (need > static_cast<int>(back)) return n - back;  // lead byte waiting for more
    return n;
  }
  return n;
}

}  // namespace ps::utf8
