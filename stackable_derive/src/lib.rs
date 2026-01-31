#![allow(missing_docs)]
//! Derive macro for the `Stackable` trait.
//!
//! This crate provides a procedural macro to automatically implement the `Stackable` trait
//! for structs with named fields. The `Stackable` trait allows batching multiple instances
//! of a type by stacking their fields.
//!
//! # Example
//!
//! ```ignore
//! use stackable_derive::Stackable;
//!
//! #[derive(Stackable)]
//! struct MyData {
//!     values: Vec<f32>,
//!     labels: Vec<i32>,
//! }
//! ```

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Derives the `Stackable` trait for a struct.
///
/// The generated implementation will create a batched version of the struct by
/// stacking each field using their respective `Stackable` implementations.
///
/// # Panics
///
/// Panics if applied to non-struct types or structs without named fields.
#[proc_macro_derive(Stackable)]
pub fn derive_stackable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    match input.data {
        Data::Struct(data) => derive_struct(name, data),
        _ => panic!("Stackable can only be derived for structs"),
    }
}

fn derive_struct(name: syn::Ident, data: syn::DataStruct) -> TokenStream {
    let fields: Vec<_> = match data.fields {
        Fields::Named(f) => f.named.into_iter().collect(),
        _ => panic!("Only named fields supported"),
    };

    // field names
    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    // 内部 Vec 変数名を生成（衝突しない）
    let vec_names: Vec<_> = field_names
        .iter()
        .map(|f| format_ident!("__stack_{}", f))
        .collect();

    let expanded = quote! {
        impl Stackable for #name
        where
            #( #field_types: Stackable<Output = #field_types> ),*
        {
            type Output = Self;

            fn stack(batch: Vec<Self>) -> Self::Output {
                #( let mut #vec_names = Vec::new(); )*

                for item in batch {
                    #( #vec_names.push(item.#field_names); )*
                }

                #name {
                    #( #field_names: <#field_types as Stackable>::stack(#vec_names) ),*
                }
            }
        }
    };

    expanded.into()
}
