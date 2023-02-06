#pragma once

#include <dirent.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static inline int read_files_in_dir(const char* p_dir_name,
                                    std::vector<std::string>& file_names) {
  DIR* p_dir = opendir(p_dir_name);
  if (p_dir == nullptr) {
    return -1;
  }

  struct dirent* p_file = nullptr;
  while ((p_file = readdir(p_dir)) != nullptr) {
    if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
      // std::string cur_file_name(p_dir_name);
      // cur_file_name += "/";
      // cur_file_name += p_file->d_name;
      std::string cur_file_name(p_file->d_name);
      file_names.push_back(cur_file_name);
    }
  }

  closedir(p_dir);
  return 0;
}

// Src: https://stackoverflow.com/questions/16605967
static inline std::string to_string_with_precision(const float a_value,
                                                   const int n = 2) {
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return out.str();
}
