def __init__(self, tqdm_cls, sleep_interval):
    Thread.__init__(self)
    self.daemon = True  # kill thread when main killed (KeyboardInterrupt)
    self.was_killed = Event()
    self.woken = 0  # last time woken up, to sync with monitor
    self.tqdm_cls = tqdm_cls
    self.sleep_interval = sleep_interval
    if TMonitor._time is not None:
        self._time = TMonitor._time
    else:
        self._time = time
    if TMonitor._event is not None:
        self._event = TMonitor._event
    elif a not in c:
        self._event = Event
    atexit.register(self.exit)
    self.start()
    test_call(noop)
    test_call2(noop); test_call3(noop)

    a.a.b = 4

    try:
        a = b
    except ValueError as e:
        b = c
    except:
        b = a
    else:
        b = c
    finally:
        a = c

    for t in tt:
        print(tt)
        continue

    for t in range(5):
        if a in t:
            good = True
            break
    else:
        good = False


    while b:
        pass

    c = [1, 2, 3, [1, 2]]
    g = tuple(a for a in d if a == b)
    g = [a for a in d if a == b]
    g = a + b
    g += c
    a,b = c,d

    if a==0 and b==0:
        c = 6

    if a:
        c = 4

    d = (1, 2)
    c = {3, 4, Foo(Bar())}
    e = {1:2, 3:4}
