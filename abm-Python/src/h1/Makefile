CXX 		:= g++ -std=c++14
EXEC 		:= main.cpp
TARGET 	:= ${EXEC:.cpp=}
SRCS 		:= $(wildcard *.cpp) 
SRCS    := $(filter-out $(EXEC), $(SRCS))
HEADERS := $(wildcard *.h)
OBJS    := ${SRCS:.cpp=.o}
CFLAGS  := -O3

DEPDIR := .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td


# COMPILE = $(CXX) $(DEPFLAGS) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d


${TARGET} : ${OBJS} ${HEADERS} ${EXEC} Makefile
	${CXX} ${CFLAGS} $(OBJS) ${EXEC} -o $@ ${LDFLAGS} 

%.o : %.C
%.o : %.C $(DEPDIR)/%.d
	$(CXX) ${CFLAGS} $<
	# $(COMPILE) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))


.PHONY: all clean tar debug print 

debug::
	$(MAKE) CFLAGS="-g -O0 -Wall -Wno-sign-compare -DDEBUG "

print:: 
	@echo ${CFLAGS}
	@echo ${LDFLAGS}
	@echo ${SRCS}

all:: ${TARGET}  

clean:: 
	rm -rf *~ ${OBJS} ${TARGET} ${TARGET}.o ${DEPDIR}

dist::
	tar cvfz ${TARGET}.tar.gz ${EXEC} ${SRCS} ${HEADERS} Makefile

