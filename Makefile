.PHONY: setup setup-musicgen worker-serve cli-run lint test fmt ios-run ios-test ensure-xcode ensure-ios-simulator client-setup client-run-desktop client-test client-build-desktop

UV_CACHE_DIR := .uv/cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) uv
UV_RUN := $(UV) run --project worker
IOS_SIMULATOR ?= iPhone 17
XCODE_DEVELOPER ?= $(shell xcode-select -p 2>/dev/null)
XCODE_APP ?= /Applications/Xcode.app/Contents/Developer
IOS_TEAM_ID ?= $(TIMBRE_IOS_TEAM_ID)
IOS_CODE_SIGN_FLAGS := DEVELOPMENT_TEAM="$(IOS_TEAM_ID)" CODE_SIGN_IDENTITY="" CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO

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

client-setup:
	cd client && yarn install --check-files

client-run-desktop:
	cd client && yarn dev:desktop:electron

client-test:
	cd client && yarn test

client-build-desktop:
	cd client && yarn build:desktop

ios-run:
	$(MAKE) ensure-ios-simulator
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcrun simctl boot "$(IOS_SIMULATOR)" || true
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcodebuild -project ios/TimbreMobile.xcodeproj -scheme TimbreMobile -destination 'platform=iOS Simulator,name=$(IOS_SIMULATOR)' -configuration Debug -derivedDataPath ios/DerivedData $(IOS_CODE_SIGN_FLAGS) build
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcrun simctl install "$(IOS_SIMULATOR)" ios/DerivedData/Build/Products/Debug-iphonesimulator/TimbreMobile.app
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcrun simctl launch "$(IOS_SIMULATOR)" com.timbre.mobile || true

ios-test:
	$(MAKE) ensure-ios-simulator
	DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcodebuild -project ios/TimbreMobile.xcodeproj -scheme TimbreMobile -destination 'platform=iOS Simulator,name=$(IOS_SIMULATOR)' -configuration Debug -derivedDataPath ios/DerivedData $(IOS_CODE_SIGN_FLAGS) test

ensure-xcode:
	@if [ -z "$(XCODE_DEVELOPER)" ] || echo "$(XCODE_DEVELOPER)" | grep -q "CommandLineTools"; then \
		echo "xcodebuild is pointing at Command Line Tools. Switch to full Xcode with:"; \
		echo "  sudo xcode-select --switch \"$(XCODE_APP)\""; \
		echo "or override with: XCODE_DEVELOPER=\"/path/to/Xcode.app/Contents/Developer\" make ios-test"; \
		exit 1; \
	fi

ensure-ios-simulator:
	$(MAKE) ensure-xcode
	@set -e; \
	AVAILABLE_OUTPUT=$$(DEVELOPER_DIR="$(XCODE_DEVELOPER)" xcrun simctl list devices available 2>&1); \
	STATUS=$$?; \
	if [ "$$STATUS" -ne 0 ]; then \
		echo "xcrun simctl failed (likely no iOS Simulator runtime installed). Install one with:"; \
		echo "  DEVELOPER_DIR=\"$(XCODE_DEVELOPER)\" xcodebuild -downloadPlatform iOS"; \
		echo "Raw simctl output:"; \
		echo "$$AVAILABLE_OUTPUT"; \
		exit "$$STATUS"; \
	fi; \
	if [ -z "$$AVAILABLE_OUTPUT" ]; then \
		echo "No iOS simulator runtimes are installed. Install one with:"; \
		echo "  DEVELOPER_DIR=\"$(XCODE_DEVELOPER)\" xcodebuild -downloadPlatform iOS"; \
		exit 1; \
	fi; \
	if ! echo "$$AVAILABLE_OUTPUT" | grep -q "$(IOS_SIMULATOR) ("; then \
		echo "Simulator '$(IOS_SIMULATOR)' was not found. Available devices:"; \
		echo "$$AVAILABLE_OUTPUT"; \
		echo "Set IOS_SIMULATOR to one of the above (e.g., IOS_SIMULATOR=\"iPhone 16 Pro\") or install a new runtime in Xcode."; \
		exit 1; \
	fi
