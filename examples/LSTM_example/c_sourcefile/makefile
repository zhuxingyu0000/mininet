CC=g++
OBJS=main.o\
	 tensor_interface.o\
	 LSTM.o\
	 basic.o\
	 activation_function.o\
	 net.o

main:$(OBJS)

%.o:%.c
	$(CC) -c -g $^ -o $@

clean:
	rm -rf $(OBJS)