pub mod deepl;
pub mod deeplx;
pub mod google;
pub mod hcfy;
pub mod imme;
pub mod kiss;

use std::borrow::Cow;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Deserializer that falls back to `Default::default()` when the JSON value is `null`.
pub fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

/// Parse and canonicalize an incoming source language parameter.
/// If absent, empty, or "auto" (case-insensitive), returns `"auto"`.
#[inline]
pub fn parse_source_lang<'a>(lang: Option<&'a str>) -> Cow<'a, str> {
    match lang.map(str::trim) {
        None | Some("") => Cow::Borrowed("auto"),
        Some(s) if s.eq_ignore_ascii_case("auto") => Cow::Borrowed("auto"),
        Some(s) => crate::utils::lang::normalize_language_code_cow(s),
    }
}

/// Parse and canonicalize an incoming target language parameter.
/// Returns an `ApiError::BadRequest` if the normalized language code is empty.
#[inline]
pub fn parse_target_lang<'a>(
    lang: &'a str,
    field_name: &str,
) -> Result<Cow<'a, str>, crate::api::ApiError> {
    let target = crate::utils::lang::normalize_language_code_cow(lang.trim());
    if target.is_empty() {
        return Err(crate::api::ApiError::BadRequest(format!(
            "Missing '{field_name}' language"
        )));
    }
    Ok(target)
}
/// A typed container for a field that can be either a single element or a sequence of elements.
///
/// Avoids intermediate JSON AST construction (`serde_json::Value`), deserializing directly
/// into either a single item or a `Vec<T>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    /// Borrow the contents as a continuous slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            OneOrMany::One(item) => std::slice::from_ref(item),
            OneOrMany::Many(items) => items.as_slice(),
        }
    }

    /// Convert the container into an owned `Vec<T>`.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        match self {
            OneOrMany::One(item) => vec![item],
            OneOrMany::Many(items) => items,
        }
    }

    /// Return the number of elements contained.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            OneOrMany::One(_) => 1,
            OneOrMany::Many(items) => items.len(),
        }
    }

    /// Return whether this container has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            OneOrMany::One(_) => false,
            OneOrMany::Many(items) => items.is_empty(),
        }
    }
}

impl<T> From<T> for OneOrMany<T> {
    #[inline]
    fn from(val: T) -> Self {
        OneOrMany::One(val)
    }
}

impl<T> From<Vec<T>> for OneOrMany<T> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        OneOrMany::Many(vec)
    }
}

impl<T> AsRef<[T]> for OneOrMany<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Serialize> Serialize for OneOrMany<T> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            OneOrMany::One(item) => item.serialize(serializer),
            OneOrMany::Many(items) => items.serialize(serializer),
        }
    }
}

impl<'de, T> Deserialize<'de> for OneOrMany<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OneOrManyVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> serde::de::Visitor<'de> for OneOrManyVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = OneOrMany<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a single value or an array of values")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                T::deserialize(serde::de::IntoDeserializer::into_deserializer(value))
                    .map(OneOrMany::One)
            }

            fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                T::deserialize(serde::de::IntoDeserializer::into_deserializer(value))
                    .map(OneOrMany::One)
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                T::deserialize(serde::de::IntoDeserializer::into_deserializer(value))
                    .map(OneOrMany::One)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(OneOrMany::Many(items))
            }
        }

        deserializer.deserialize_any(OneOrManyVisitor(std::marker::PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_or_many_deserialization() {
        let single: OneOrMany<String> = serde_json::from_str(r#""hello""#).unwrap();
        assert_eq!(single, OneOrMany::One("hello".to_string()));
        assert_eq!(single.as_slice(), &["hello"]);
        assert_eq!(single.len(), 1);
        assert!(!single.is_empty());

        let multi: OneOrMany<String> = serde_json::from_str(r#"["hello","world"]"#).unwrap();
        assert_eq!(
            multi,
            OneOrMany::Many(vec!["hello".to_string(), "world".to_string()])
        );
        assert_eq!(multi.as_slice(), &["hello", "world"]);
        assert_eq!(multi.len(), 2);

        let empty: OneOrMany<String> = serde_json::from_str(r#"[]"#).unwrap();
        assert_eq!(empty, OneOrMany::Many(vec![]));
        assert!(empty.is_empty());

        let invalid_int: Result<OneOrMany<String>, _> = serde_json::from_str(r#"123"#);
        assert!(invalid_int.is_err());

        let invalid_elem: Result<OneOrMany<String>, _> = serde_json::from_str(r#"["hello", 123]"#);
        assert!(invalid_elem.is_err());
    }

    #[test]
    fn test_one_or_many_serialization() {
        let single = OneOrMany::One("hello".to_string());
        assert_eq!(serde_json::to_string(&single).unwrap(), r#""hello""#);

        let multi = OneOrMany::Many(vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(
            serde_json::to_string(&multi).unwrap(),
            r#"["hello","world"]"#
        );
    }
}
