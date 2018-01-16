import numpy
import functools

class CheckError(Exception):
    def __init__(self,*args,**kw):
        self.fun = kw.get("fun","?")
        self.var = kw.get("var","?")
        self.description = " ".join([strc(x) for x in args])

    def __str__(self):
        result = "\nCheckError for argument "
        result += str(self.var)
        result += " of function "
        result += str(self.fun)
        result += "\n"
        result += self.description
        return result

class CheckWarning(CheckError):
    def __init__(self,*args,**kw):
        self.fun = kw.get("fun","?")
        self.var = kw.get("var","?")
        self.description = " ".join([strc(x) for x in args])

    def __str__(self):
        result = "\nCheckWarning for argument "
        result += str(self.var)
        result += " of function "
        result += str(self.fun)
        result += "\n"
        result += self.description
        result += "(This can happen occasionally during normal operations and isn't necessarily a bug or problem.)\n"
        return result

def strc(   arg,
            n=10    ):
    if isinstance(arg,float):
        return "%.3g"%arg
    if type(arg)==list:
        return "[%s|%d]"%(",".join([strc(x) for x in arg[:3]]),len(arg))
    if type(arg)==numpy.ndarray:
        return "<ndarray-%x %s %s [%s,%s]>"%(id(arg),arg.shape,str(arg.dtype),numpy.amin(arg),numpy.amax(arg))
    return str(arg).replace("\n"," ")


def makeargcheck(   message,
                    warning=0   ):
    def decorator(f):
        def wrapper(arg):
            if not f(arg):
                if warning:
                    raise CheckWarning(strc(arg)+" of type "+str(type(arg))+": "+str(message))
                else:
                    raise CheckError(strc(arg)+" of type "+str(type(arg))+": "+str(message))
        return wrapper
    return decorator

def checktype(  value,
                type_   ):
    if type_ is True:
        return value

    if type(type_)==type:
        if not isinstance(value,type_):
            raise CheckError("isinstance failed",value,"of type",type(value),"is not of type",type_)
        return value

    if type(type_)==list:
        if not numpy.iterable(value):
            raise CheckError("expected iterable",value)
        for x in value:
            if not reduce(max,[isinstance(x,t) for t in type_]):
                raise CheckError("element",x,"of type",type(x),"fails to be of type",type_)
        return value

    if type(type_)==set:
        for t in type_:
            if isinstance(value,t): return value
        raise CheckError("set membership failed",value,type_,var=var) 

    if type(type_)==tuple:
        for t in type_:
            checktype(value,type_)
        return value

    if callable(type_):
        type_(value)
        return value

    raise Exception("unknown type spec: %s"%type_)

def checks( *types,
            **ktypes    ):
    
    def argument_check_decorator(f):
        @functools.wraps(f)
        def argument_checks(*args,**kw):
            name                = f.func_name
            argnames            = f.func_code.co_varnames[:f.func_code.co_argcount]
            kw3                 = [(var,value,ktypes.get(var,True)) for var,value in kw.items()]
            for var,value,type_ in zip(argnames,args,types)+kw3:
                try:
                    checktype(value,type_)
                except AssertionError as e:
                    raise CheckError(e.message,*e.args,var=var,fun=f)
                except CheckError as e:
                    e.fun = f
                    e.var = var
                    raise e
                except:
                    raise
            result              = f(*args,**kw)
            checktype(result,kw.get("_",True))
            return result
        return argument_checks
    return argument_check_decorator