from test_scenarios import test_different_kernels, test_noise_levels


def main():
    test_different_kernels("simple.png")
    test_noise_levels("simple.png")

if __name__ == "__main__":
    main()