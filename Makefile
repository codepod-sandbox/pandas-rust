.PHONY: setup build test clean

setup:
	git config core.hooksPath .githooks

build:
	cargo build -p pandas-rust-wasm

test: build
	cargo test -p pandas-rust-core
	target/debug/pandas-python -m pytest tests/python/

clean:
	cargo clean
