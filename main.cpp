#include <chrono>
#include <iostream>

int tere(int options, bool doPrint = true);

int main(int argc, char* argv[])
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    tere(5);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Alg took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << "ms" << std::endl;

    return 0;
}