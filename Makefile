CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm
SRCDIR = source
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:.c=.o)
TARGET = rover

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(OBJECTS) $(TARGET)