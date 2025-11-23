.PHONY: setup setup-musicgen worker-serve cli-run lint test fmt ios-run ios-test ensure-xcode

UV_CACHE_DIR := .uv/cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) uv
UV_RUN := $(UV) run --project worker
IOS_SIMULATOR ?= iPhone 15
XCODE_DEVELOPER ?= $(shell xcode-select -p 2>/dev/null)
XCODE_APP ?= /Applications/Xcode.app/Contents/Developer

$(UV_CACHE_DIR):
	mkdir -p $(UV_CACHE_DIR)

setup:
	$(MAKE) $(UV_CACHE_DIR)
	$(UV) sync --project worker --extra dev --extra inference
	cargo fetch

setup-musicgen:
	$(MAKE) $(UV_CACHE_DIR)
	$(UV) sync --project worker --extra dev --extra inference
	cargo fetch

worker-serve:
	$(MAKE) $(UV_CACHE_DIR)
	$(UV_RUN) uvicorn timbre_worker.app.main:app --port 8000

worker-serve-reload:
	$(MAKE) $(UV_CACHE_DIR)
	$(UV_RUN) uvicorn timbre_worker.app.main:app --reload --port 8000

cli-run:
	cargo run -p timbre-cli

fmt:
	cargo fmt

lint:
	cargo fmt --check
	cargo clippy -- -D warnings
	$(MAKE) $(UV_CACHE_DIR)
	$(UV_RUN) ruff check
	$(UV_RUN) mypy

test:
	cargo test
	$(MAKE) $(UV_CACHE_DIR)
	$(UV_RUN) pytest

ios-run:
	$(MAKE) ensure-xcode
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcrun simctl boot "$(IOS_SIMULATOR)" || true
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcodebuild -project ios/TimbreMobile.xcodeproj -scheme TimbreMobile -destination 'platform=iOS Simulator,name=$(IOS_SIMULATOR)' -configuration Debug -derivedDataPath ios/DerivedData build
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcrun simctl install "$(IOS_SIMULATOR)" ios/DerivedData/Build/Products/Debug-iphonesimulator/TimbreMobile.app
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcrun simctl launch "$(IOS_SIMULATOR)" com.timbre.mobile || true

ios-test:
	$(MAKE) ensure-xcode
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcodebuild -project ios/TimbreMobile.xcodeproj -scheme TimbreMobile -destination 'platform=iOS Simulator,name=$(IOS_SIMULATOR)' -configuration Debug -derivedDataPath ios/DerivedData test

ensure-xcode:
	@if [ -z "$(XCODE_DEVELOPER)" ] || echo "$(XCODE_DEVELOPER)" | grep -q "CommandLineTools"; then \
		echo "xcodebuild is pointing at Command Line Tools. Switch to full Xcode with:"; \
		echo "  sudo xcode-select --switch \"$(XCODE_APP)\""; \
		echo "or override with: XCODE_DEVELOPER=\"/path/to/Xcode.app/Contents/Developer\" make ios-test"; \
		exit 1; \
	fi
