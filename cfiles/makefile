CC=g++
OBJS=main.o\
	 net.o\
	 tensor_interface.o\
	 core.o

main:$(OBJS)

%.o:%.c
	$(CC) -c $^ -o $@

clean:
	rm -rf $(OBJS)