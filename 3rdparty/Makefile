# SPIRV-Tools 
SPIRV_TOOLS_DIR?=SPIRV-Tools
SPIRV_TOOLS_EXTERNAL_DIR:=$(SPIRV_TOOLS_DIR)/external
SPIRV_TOOLS_BUILD_DIR:=$(SPIRV_TOOLS_DIR)/build

# glslang
GLSLANG_DIR?=glslang
GLSLANG_BUILD_DIR:=$(GLSLANG_DIR)/build

ifeq ($(origin CMAKE), undefined)
  CMAKE := $(shell ./find-cmake.sh)
endif
NCPUS:=$(shell sysctl -n hw.ncpu)

# Determine if the current SDK version is Big Sur or later
# 
# NOTE: 
# When running Xcode 12 on Intel hardware, the SDK returns 10.16.
# On Apple SI, SDK returns 11.0
MACOS_SDK_VERSION := $(shell xcrun --sdk macosx --show-sdk-version)

# need to check the current SDK is either >= 10.16 or 11.0
TARGET_ARCHITECTURES ?= $(shell ./version_gte.sh $(MACOS_SDK_VERSION) 11.0 || ./version_gte.sh $(MACOS_SDK_VERSION) 10.16 && echo 'arm64;x86_64' || echo 'x86_64')

# Deployment floor for back-deployment (libc++ availability macros).
# ToolDependencies passes Xcode build settings in env, so MACOSX_DEPLOYMENT_TARGET
# is set when building from Xcode; default to Boxer's minimum when running make by hand.
# Without this, CMake defaults to the host SDK (e.g. 26.2), baking in
# __libcpp_verbose_abort (needs macOS 13.3 libc++) and breaking launch on Monterey.
TARGET_DEPLOYMENT ?= $(if $(MACOSX_DEPLOYMENT_TARGET),$(MACOSX_DEPLOYMENT_TARGET),12.0)

# CMake 4.x removed compatibility with cmake_minimum_required(<3.5), which both
# bundled projects still declare. Pass a floor so clean reconfigures work with
# modern CMake. Override from env if needed: CMAKE_POLICY_FLAG=
CMAKE_POLICY_FLAG ?= -DCMAKE_POLICY_VERSION_MINIMUM=3.5

all: libSPIRV-Tools-opt.a build_info.h

install: all
.PHONY: install

# Force CMakeCache validation on every invocation. Without this, make treats a
# stale CMakeCache.txt as up-to-date (it's newer than CMakeLists.txt) and never
# reruns the configure recipes below, so a cached
# CMAKE_OSX_DEPLOYMENT_TARGET=26.2 or dead CMAKE_COMMAND persists forever.
# Declaring the cache files .PHONY makes their recipes run each time; each
# recipe exits immediately when the cache is current, otherwise it wipes its
# build dir and reconfigures. The downstream inner-make calls are cheap
# no-ops (~1-2s) when nothing changed. NOTE: the cache path intentionally does
# NOT list this Makefile as a normal prerequisite -- glslang/ and
# SPIRV-Tools/ are submodules, so editing this file updates its mtime to newer
# than their CMakeLists.txt and a normal prerequisite would force a full
# rebuild of both static libs on every Xcode build. .PHONY + early-exit
# avoids that while still catching stale caches.
.PHONY: $(GLSLANG_BUILD_DIR)/CMakeCache.txt $(SPIRV_TOOLS_BUILD_DIR)/CMakeCache.txt

clean: clean-spirv clean-glslang
.PHONY: clean

# glslang

clean-glslang:
	if [[ -e $(GLSLANG_BUILD_DIR) ]]; then $(MAKE) -C $(GLSLANG_BUILD_DIR) clean; fi
	rm -rf $(GLSLANG_BUILD_DIR)
.PHONY: clean-glslang

build_info.h: $(GLSLANG_BUILD_DIR)/include/glslang/build_info.h
.PHONY: build_info.h

$(GLSLANG_BUILD_DIR)/include/glslang/build_info.h: $(GLSLANG_DIR)/CHANGES.md $(GLSLANG_DIR)/build_info.h.tmpl $(GLSLANG_BUILD_DIR)/CMakeCache.txt
	$(MAKE) -C $(GLSLANG_BUILD_DIR) glslang-build-info/fast

$(GLSLANG_BUILD_DIR)/CMakeCache.txt: $(GLSLANG_DIR)/CMakeLists.txt
	@if [ -f $@ ]; then \
	  CACHED_DEPLOY=$$(grep '^CMAKE_OSX_DEPLOYMENT_TARGET[^=]*=' $@ 2>/dev/null | cut -d= -f2); \
	  CACHED_CMAKE=$$(grep '^CMAKE_COMMAND:INTERNAL=' $@ 2>/dev/null | cut -d= -f2-); \
	  if [ "$$CACHED_DEPLOY" = "$(TARGET_DEPLOYMENT)" ] && { [ -z "$$CACHED_CMAKE" ] || [ -x "$$CACHED_CMAKE" ] || command -v "$$CACHED_CMAKE" >/dev/null 2>&1; }; then \
	    exit 0; \
	  elif [ "$$CACHED_DEPLOY" != "$(TARGET_DEPLOYMENT)" ]; then \
	    echo "glslang: deployment target changed ($$CACHED_DEPLOY -> $(TARGET_DEPLOYMENT)), wiping $(GLSLANG_BUILD_DIR)"; \
	  else \
	    echo "glslang: cached CMake ($$CACHED_CMAKE) missing, wiping $(GLSLANG_BUILD_DIR)"; \
	  fi; \
	  rm -rf $(GLSLANG_BUILD_DIR); \
	fi; \
	mkdir -p $(GLSLANG_BUILD_DIR); \
	$(CMAKE) -B $(GLSLANG_BUILD_DIR) -S $(GLSLANG_DIR) $(CMAKE_POLICY_FLAG) -DCMAKE_OSX_DEPLOYMENT_TARGET='$(TARGET_DEPLOYMENT)' -DCMAKE_OSX_ARCHITECTURES='$(TARGET_ARCHITECTURES)'

# SPIRV-Tools

clean-spirv:
	if [[ -e $(SPIRV_TOOLS_BUILD_DIR) ]]; then $(MAKE) -C $(SPIRV_TOOLS_BUILD_DIR) clean; fi
	rm -rf $(SPIRV_TOOLS_BUILD_DIR)
	rm -rf $(SPIRV_TOOLS_EXTERNAL_DIR)/SPIRV-Headers
.PHONY : clean-spirv

libSPIRV-Tools-opt.a: $(SPIRV_TOOLS_BUILD_DIR)/source/opt/libSPIRV-Tools-opt.a
.PHONY : libSPIRV-Tools-opt.a

$(SPIRV_TOOLS_BUILD_DIR)/source/opt/libSPIRV-Tools-opt.a: $(SPIRV_TOOLS_BUILD_DIR)/CMakeCache.txt
	mkdir -p $(SPIRV_TOOLS_BUILD_DIR)
	$(MAKE) -j$(NCPUS) -C $(SPIRV_TOOLS_BUILD_DIR) SPIRV-Tools-opt

$(SPIRV_TOOLS_BUILD_DIR)/CMakeCache.txt: $(SPIRV_TOOLS_DIR)/CMakeLists.txt $(SPIRV_TOOLS_EXTERNAL_DIR)/SPIRV-Headers
	@if [ -f $@ ]; then \
	  CACHED_DEPLOY=$$(grep '^CMAKE_OSX_DEPLOYMENT_TARGET[^=]*=' $@ 2>/dev/null | cut -d= -f2); \
	  CACHED_CMAKE=$$(grep '^CMAKE_COMMAND:INTERNAL=' $@ 2>/dev/null | cut -d= -f2-); \
	  if [ "$$CACHED_DEPLOY" = "$(TARGET_DEPLOYMENT)" ] && { [ -z "$$CACHED_CMAKE" ] || [ -x "$$CACHED_CMAKE" ] || command -v "$$CACHED_CMAKE" >/dev/null 2>&1; }; then \
	    exit 0; \
	  elif [ "$$CACHED_DEPLOY" != "$(TARGET_DEPLOYMENT)" ]; then \
	    echo "SPIRV-Tools: deployment target changed ($$CACHED_DEPLOY -> $(TARGET_DEPLOYMENT)), wiping $(SPIRV_TOOLS_BUILD_DIR)"; \
	  else \
	    echo "SPIRV-Tools: cached CMake ($$CACHED_CMAKE) missing, wiping $(SPIRV_TOOLS_BUILD_DIR)"; \
	  fi; \
	  rm -rf $(SPIRV_TOOLS_BUILD_DIR); \
	fi; \
	mkdir -p $(SPIRV_TOOLS_BUILD_DIR); \
	$(CMAKE) -B $(SPIRV_TOOLS_BUILD_DIR) -S $(SPIRV_TOOLS_DIR) $(CMAKE_POLICY_FLAG) -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_OSX_ARCHITECTURES='$(TARGET_ARCHITECTURES)' -DCMAKE_OSX_DEPLOYMENT_TARGET='$(TARGET_DEPLOYMENT)' -DSPIRV_WERROR=OFF

$(SPIRV_TOOLS_EXTERNAL_DIR)/SPIRV-Headers:
	pushd $(SPIRV_TOOLS_DIR); \
	git clone https://github.com/KhronosGroup/SPIRV-Headers.git external/SPIRV-Headers; \
	cd external/SPIRV-Headers; \
	git checkout 1.5.4; \
	popd

print:
	@echo '$(TARGET_ARCHITECTURES)'
