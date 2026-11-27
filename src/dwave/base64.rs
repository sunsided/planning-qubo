//! Standard base64 (RFC 4648, with padding), as SAPI uses for its binary arrays.

use alloc::string::String;
use alloc::vec::Vec;

/// The standard base64 alphabet.
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes `bytes` as padded base64.
pub(super) fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);

    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let word = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);

        for k in 0..4 {
            if k <= chunk.len() {
                out.push(char::from(ALPHABET[(word >> (18 - 6 * k) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }

    out
}

/// Decodes padded base64, or returns `None` for a wrong length, character or padding.
pub(super) fn decode(text: &str) -> Option<Vec<u8>> {
    let text = text.as_bytes();

    if text.len() % 4 != 0 {
        return None;
    }

    let mut out = Vec::with_capacity(text.len() / 4 * 3);

    for (index, quad) in text.chunks(4).enumerate() {
        let last = index + 1 == text.len() / 4;
        let padding = quad.iter().rev().take_while(|&&c| c == b'=').count();

        if padding > 2 || (padding > 0 && !last) {
            return None;
        }

        let mut word = 0_u32;

        for &c in &quad[..4 - padding] {
            word = word << 6 | u32::from(sextet(c)?);
        }

        word <<= 6 * u32::try_from(padding).ok()?;
        let bytes = word.to_be_bytes();
        out.extend_from_slice(&bytes[1..4 - padding]);
    }

    Some(out)
}

/// The value of one base64 character.
fn sextet(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Unit tests against the RFC 4648 test vectors.
#[cfg(test)]
mod tests {
    use super::{decode, encode};

    /// The RFC 4648 section 10 vectors round-trip.
    #[test]
    fn rfc_vectors() {
        let vectors = [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];

        for (plain, coded) in vectors {
            assert_eq!(encode(plain.as_bytes()), coded);
            assert_eq!(decode(coded).as_deref(), Some(plain.as_bytes()));
        }
    }

    /// Malformed input is rejected.
    #[test]
    fn rejects_malformed_input() {
        assert_eq!(decode("Zg="), None);
        assert_eq!(decode("Zg=a"), None);
        assert_eq!(decode("Z==="), None);
        assert_eq!(decode("Zg==Zg=="), None);
        assert_eq!(decode("Zm9*"), None);
    }
}
