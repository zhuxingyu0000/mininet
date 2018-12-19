CC=g++
OBJS=main.o\
	 net.o\
	 tensor_interface.o\
	 core.o\
	 activation_function.o

main:$(OBJS)

%.o:%.c
	$(CC) -c $^ -o $@

clean:
	rm -rf $(OBJS)