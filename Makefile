CXX = g++

CXXFLAGS = -Wall -std=c++17 -O3 -march=native -DNDEBUG -fopenmp \
           -I include -I /usr/include/eigen3 -MMD -MP

LDFLAGS  = -fopenmp
LDLIBS   = -lopenblas -llapack

SRC_DIR = src
BUILD_DIR = build
BIN = out

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN)

-include $(OBJS:.o=.d)

.PHONY: all clean

