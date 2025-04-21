#pragma once
#include <fstream>
#include <string>
#include <iomanip>

class CSVLogger {
public:
    explicit CSVLogger(const std::string& fname)
        : file(fname, std::ios::out)
    {
        file << "tag,gen,idx,maxIter,fpsErr,gpuTimeMs,boundary,density,rank\n";
    }
    ~CSVLogger() { file.close(); }

    template<typename... Args>
    void row(const std::string& tag, int gen, Args&&... xs)
    {
        file << tag << ',' << gen;
        ((file << ',' << xs), ...);
        file << '\n';
    }
private:
    std::ofstream file;
};
