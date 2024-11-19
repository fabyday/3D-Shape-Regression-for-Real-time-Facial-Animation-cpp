#include "dataset.h"
#include <filesystem>
#include <utility>
#include <map>
#include <fstream>
#include "string_function.h"
#include "cnpy.h"
#include <iostream>
void Dataset::load(const std::string& pth)
{





	std::filesystem::path root = pth;
	std::map<std::string, std::string> mm;

	std::filesystem::path meta_path = "";
	for (auto const& dir_entry : std::filesystem::directory_iterator{ root }) {
		if (dir_entry.path().filename() == "meta.txt")
			meta_path = dir_entry.path();
	}
	if (meta_path == "")
		throw std::exception("meta.txt not found.");


	std::ifstream  in(meta_path.string());
	std::string buf;
	while (std::getline(in, buf)) {
		int pos = buf.find(":", 0);
		std::string key = buf.substr(0, pos);
		std::string value = buf.substr(pos + 1, buf.size());
		trim(key);
		trim(value);
		mm.insert(std::pair(key, value));
	}
	std::string image_extension = mm["image_extension"];
	std::string image_root_location= mm["image_root_location"];
	std::string  S_Rtinv_index_list_path= mm["S_Rtinv_index_list"];
	std::string image_name_lication = mm["image_name_location"];
	std::string  Rt_inv_location  = mm["Rt_inv_location"];
	std::string S_init_location = mm["S_init_location"];
	std::string S_location  = mm["S_location"];
	std::string S_original_location  = mm["S_original_location"];
	std::string root_path = mm["data_root"];


	std::filesystem::path data_root = pth;
	data_root /= root_path;
	cnpy::NpyArray S_Rtinv_index_list_data = cnpy::npy_load((data_root / S_Rtinv_index_list_path).string() );
	cnpy::NpyArray Ss_data = cnpy::npy_load((data_root / S_location).string() );
	cnpy::NpyArray S_original_data = cnpy::npy_load((data_root / S_original_location).string() );
	cnpy::NpyArray S_init_data = cnpy::npy_load((data_root / S_init_location).string() );
	cnpy::NpyArray Rt_inv_data = cnpy::npy_load((data_root / Rt_inv_location).string() );


	using F = float;
	using D = double;
	using MatrixRf = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatrixRd = Eigen::Matrix<D, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	int Rt_row = Rt_inv_data.shape[1];
	int Rt_col = Rt_inv_data.shape[2];
	for (int i = 0; i < Rt_inv_data.shape[0]; i++) {
		m_Rt_inv_list.emplace_back(Eigen::Map<MatrixRf>(Rt_inv_data.data<float>() + i * Rt_row * Rt_col, Rt_row, Rt_col).cast<double>());
	}

	//std::cout << m_Rt_inv_list[0] << std::endl << m_Rt_inv_list[1] << std::endl;

	int ss_row = Ss_data.shape[1];
	int ss_col = Ss_data.shape[2];
	for (int i = 0; i < Ss_data.shape[0]; i++) {
		m_Ss_list.emplace_back(Eigen::Map<MatrixRd>(Ss_data.data<double>() + i * ss_row * ss_col, ss_row, ss_col));
	}

	//std::cout << m_Ss_list[0] << std::endl <<std::endl << m_Ss_list[1] << std::endl;

	for (int i = 0; i < Ss_data.shape[0]; i++) {
		m_S_init_list.emplace_back(*(S_init_data.data<int>() + i ));
	}
	//std::cout << m_S_init_list[0] << std::endl <<std::endl << m_S_init_list[1] << std::endl;

	int so_row = S_original_data.shape[1];
	int so_col = S_original_data.shape[2];
	for (int i = 0; i < S_original_data.shape[0]; i++) {
		m_S_original_list.emplace_back(Eigen::Map<MatrixRd>(S_original_data.data<double>() + i * so_row * so_col, so_row, so_col));
	}
	//std::cout << m_S_original_list[0] << std::endl <<std::endl << m_S_original_list[1] << std::endl;



	for (int i = 0; i < S_Rtinv_index_list_data.shape[0]; i++) {
		m_SRt_index_list.emplace_back( (int)*(S_Rtinv_index_list_data.data< unsigned int >() + i ) );
	}

	//std::cout << m_SRt_index_list[m_SRt_index_list.size()-1] << std::endl;

	

}
