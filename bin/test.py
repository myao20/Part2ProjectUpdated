import time


def main():
    print("Hello world")
    start = time.time()
    end = time.time()
    print(f"{(end - start) / 60:.3f} minutes")


if __name__ == "__main__":
    main()
