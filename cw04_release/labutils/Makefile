# Alternative GNU Make project makefile autogenerated by Premake

ifndef config
  config=debug_x64
endif

ifndef verbose
  SILENT = @
endif

.PHONY: clean prebuild

SHELLTYPE := posix
ifeq (.exe,$(findstring .exe,$(ComSpec)))
	SHELLTYPE := msdos
endif

# Configurations
# #############################################

RESCOMP = windres
INCLUDES += -I../third_party/volk/include -I../third_party/vulkan/include -I../third_party/stb/include -I../third_party/glfw/include -I../third_party/VulkanMemoryAllocator/include -I../third_party/glm/include -I../third_party/rapidobj/include -I../third_party/tgen/include -I../third_party/zstd/include
FORCE_INCLUDE +=
ALL_CPPFLAGS += $(CPPFLAGS) -MD -MP $(DEFINES) $(INCLUDES)
ALL_RESFLAGS += $(RESFLAGS) $(DEFINES) $(INCLUDES)
LIBS += -ldl
LDDEPS +=
LINKCMD = $(AR) -rcs "$@" $(OBJECTS)
define PREBUILDCMDS
endef
define PRELINKCMDS
endef
define POSTBUILDCMDS
endef

ifeq ($(config),debug_x64)
TARGETDIR = ../lib
TARGET = $(TARGETDIR)/liblabutils-debug-x64-gcc.a
OBJDIR = ../_build_/debug-x64-gcc/x64/debug/labutils
DEFINES += -D_DEBUG=1 -DGLM_FORCE_RADIANS=1 -DGLM_FORCE_SIZE_T_LENGTH=1 -DZSTD_DISABLE_ASM=1
ALL_CFLAGS += $(CFLAGS) $(ALL_CPPFLAGS) -m64 -g -march=native -Wall -pthread
ALL_CXXFLAGS += $(CXXFLAGS) $(ALL_CPPFLAGS) -m64 -g -std=c++17 -march=native -Wall -pthread
ALL_LDFLAGS += $(LDFLAGS) -L/usr/lib64 -m64 -pthread

else ifeq ($(config),release_x64)
TARGETDIR = ../lib
TARGET = $(TARGETDIR)/liblabutils-release-x64-gcc.a
OBJDIR = ../_build_/release-x64-gcc/x64/release/labutils
DEFINES += -DNDEBUG=1 -DGLM_FORCE_RADIANS=1 -DGLM_FORCE_SIZE_T_LENGTH=1 -DZSTD_DISABLE_ASM=1
ALL_CFLAGS += $(CFLAGS) $(ALL_CPPFLAGS) -m64 -O2 -march=native -Wall -pthread
ALL_CXXFLAGS += $(CXXFLAGS) $(ALL_CPPFLAGS) -m64 -O2 -std=c++17 -march=native -Wall -pthread
ALL_LDFLAGS += $(LDFLAGS) -L/usr/lib64 -m64 -s -pthread

endif

# Per File Configurations
# #############################################


# File sets
# #############################################

GENERATED :=
OBJECTS :=

GENERATED += $(OBJDIR)/allocator.o
GENERATED += $(OBJDIR)/context_helpers.o
GENERATED += $(OBJDIR)/error.o
GENERATED += $(OBJDIR)/to_string.o
GENERATED += $(OBJDIR)/vkbuffer.o
GENERATED += $(OBJDIR)/vkimage.o
GENERATED += $(OBJDIR)/vkobject.o
GENERATED += $(OBJDIR)/vkutil.o
GENERATED += $(OBJDIR)/vulkan_context.o
GENERATED += $(OBJDIR)/vulkan_window.o
OBJECTS += $(OBJDIR)/allocator.o
OBJECTS += $(OBJDIR)/context_helpers.o
OBJECTS += $(OBJDIR)/error.o
OBJECTS += $(OBJDIR)/to_string.o
OBJECTS += $(OBJDIR)/vkbuffer.o
OBJECTS += $(OBJDIR)/vkimage.o
OBJECTS += $(OBJDIR)/vkobject.o
OBJECTS += $(OBJDIR)/vkutil.o
OBJECTS += $(OBJDIR)/vulkan_context.o
OBJECTS += $(OBJDIR)/vulkan_window.o

# Rules
# #############################################

all: $(TARGET)
	@:

$(TARGET): $(GENERATED) $(OBJECTS) $(LDDEPS) | $(TARGETDIR)
	$(PRELINKCMDS)
	@echo Linking labutils
	$(SILENT) $(LINKCMD)
	$(POSTBUILDCMDS)

$(TARGETDIR):
	@echo Creating $(TARGETDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) mkdir -p $(TARGETDIR)
else
	$(SILENT) mkdir $(subst /,\\,$(TARGETDIR))
endif

$(OBJDIR):
	@echo Creating $(OBJDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) mkdir -p $(OBJDIR)
else
	$(SILENT) mkdir $(subst /,\\,$(OBJDIR))
endif

clean:
	@echo Cleaning labutils
ifeq (posix,$(SHELLTYPE))
	$(SILENT) rm -f  $(TARGET)
	$(SILENT) rm -rf $(GENERATED)
	$(SILENT) rm -rf $(OBJDIR)
else
	$(SILENT) if exist $(subst /,\\,$(TARGET)) del $(subst /,\\,$(TARGET))
	$(SILENT) if exist $(subst /,\\,$(GENERATED)) del /s /q $(subst /,\\,$(GENERATED))
	$(SILENT) if exist $(subst /,\\,$(OBJDIR)) rmdir /s /q $(subst /,\\,$(OBJDIR))
endif

prebuild: | $(OBJDIR)
	$(PREBUILDCMDS)

ifneq (,$(PCH))
$(OBJECTS): $(GCH) | $(PCH_PLACEHOLDER)
$(GCH): $(PCH) | prebuild
	@echo $(notdir $<)
	$(SILENT) $(CXX) -x c++-header $(ALL_CXXFLAGS) -o "$@" -MF "$(@:%.gch=%.d)" -c "$<"
$(PCH_PLACEHOLDER): $(GCH) | $(OBJDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) touch "$@"
else
	$(SILENT) echo $null >> "$@"
endif
else
$(OBJECTS): | prebuild
endif


# File Rules
# #############################################

$(OBJDIR)/allocator.o: allocator.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/context_helpers.o: context_helpers.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/error.o: error.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/to_string.o: to_string.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/vkbuffer.o: vkbuffer.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/vkimage.o: vkimage.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/vkobject.o: vkobject.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/vkutil.o: vkutil.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/vulkan_context.o: vulkan_context.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"
$(OBJDIR)/vulkan_window.o: vulkan_window.cpp
	@echo "$(notdir $<)"
	$(SILENT) $(CXX) $(ALL_CXXFLAGS) $(FORCE_INCLUDE) -o "$@" -MF "$(@:%.o=%.d)" -c "$<"

-include $(OBJECTS:%.o=%.d)
ifneq (,$(PCH))
  -include $(PCH_PLACEHOLDER).d
endif