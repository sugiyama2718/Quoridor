# Makefile

# コンパイラ
CXX = g++

# コンパイルオプション
CXXFLAGS = -fPIC -Wall -Wextra

# 共有ライブラリを作成するためのオプション
LDFLAGS = -shared

# ターゲット共有ライブラリ
# TARGET =  State_util.so  # Linux
TARGET =  State_util.dll  # Windows

# ソースファイル
SRCS = State_util.cpp Search_util.cpp

# オブジェクトファイル（ソースファイルから.oファイルに変換）
OBJS = $(SRCS:.cpp=.o)

# デフォルトターゲット
all: $(TARGET)

# 共有ライブラリの作成ルール
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# .cppファイルから.oファイルを作成するルール
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# クリーンアップ
clean:
	rm -f $(OBJS) $(TARGET)
