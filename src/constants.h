#pragma once 



#include <vector>
#include <string>
#include "cnpy.h"
inline void get_inner_contour(const std::string& name, std::vector<int>& list) {

	cnpy::NpyArray arr_data = cnpy::npy_load(name);
	
	list = arr_data.as_vec<int>();

}

