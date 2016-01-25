all:
	for filename in *py; do \
		python $$filename ; \
	done
clean: 
	rm *png
