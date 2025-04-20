# Compilatore e flag
CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -O3 -march=native -I /usr/include/eigen3 -DNDEBUG -I include

# Cartelle
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN = wd.o

# Trova tutti i file .cpp nel progetto (ricorsivamente)
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Regola principale
all: $(BIN)

# Regola per creare l'eseguibile
$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Regola per compilare ogni .cpp in .o (output nella cartella build/)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Pulizia
clean:
	rm -rf $(BUILD_DIR) $(BIN)

.PHONY: all clean
