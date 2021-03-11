from invoke import task

@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")
    print("Done cleaning")


@task()
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")
    print("Done building")


@task(pre=[build], post=[delete])
def run(c, k, n, Random=True):
    print("start running")
    if Random:
        c.run("python3.8.5 main.py {} {}".format(k, n))
    else:
        c.run("python3.8.5 main.py {} {} --Random".format(k, n))
