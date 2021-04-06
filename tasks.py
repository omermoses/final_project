from invoke import task

@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")
    print("Done cleaning")


@task()
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")
    print("Done building")


@task(pre=[build], post=[delete], optional=['k','n'])
def run(c, k=-1, n=-1, Random=True):
    """
        run calls build, then runs the main program, and after calls delete.
        default value of k,n is -1

    """

    print("start running")
    if Random:
        c.run("python3.8.5 main.py {} {}".format(k, n))
    else:
        c.run("python3.8.5 main.py {} {} --Random".format(k, n))
        # c.run("python3.8.5 main.py {} {} --Random {}".format(k, n, text))





