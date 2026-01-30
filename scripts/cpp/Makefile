CXX = g++
CXXFLAGS = -Wall -std=c++17 -fopenmp -O3 -march=native -DNDEBUG -I /usr/include/eigen3 -lopenblas -llapack -I include

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN = out.o

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN)

.PHONY: all clean
