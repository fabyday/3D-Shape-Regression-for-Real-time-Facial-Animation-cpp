
#pragma once 
#include <eigen/Core>
#include <string>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <filesystem>

inline bool compareNat(const std::string& a, const std::string& b)
{
	if (a.empty())
		return true;
	if (b.empty())
		return false;
	if (std::isdigit(a[0]) && !std::isdigit(b[0]))
		return true;
	if (!std::isdigit(a[0]) && std::isdigit(b[0]))
		return false;
	if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
	{
		if (std::toupper(a[0]) == std::toupper(b[0]))
			return compareNat(a.substr(1), b.substr(1));
		return (std::toupper(a[0]) < std::toupper(b[0]));
	}

	// Both strings begin with digit --> parse both numbers
	std::istringstream issa(a);
	std::istringstream issb(b);
	int ia, ib;
	issa >> ia;
	issb >> ib;
	if (ia != ib)
		return ia < ib;

	// Numbers are the same --> remove numbers and recurse
	std::string anew, bnew;
	std::getline(issa, anew);
	std::getline(issb, bnew);
	return (compareNat(anew, bnew));
}
struct Blendshapes{



	mutable Eigen::MatrixXd neutral_pose;
	//Eigen::MatrixXd pose;
	std::vector<Eigen::MatrixXd> pose;
	Eigen::MatrixXd flaten_pose;
	Eigen::MatrixXi F;

	void compile() {
		int row_size = pose[0].size();
		int col_size = pose.size();
		Eigen::Map<Eigen::MatrixXd> flat_neutral_pose(neutral_pose.data(), neutral_pose.size(), 1);
		flaten_pose.conservativeResize(row_size, col_size);
		for (int i = 0; i < col_size; i++) {
			//flaten_pose.block(i,0, row_size, 1) = flat_neutral_pose - Eigen::Map<Eigen::MatrixXd>(pose[i].data(), row_size, 1);
			//flaten_pose.block(0, i, row_size, 1) = flat_neutral_pose - Eigen::Map<Eigen::MatrixXd>(pose[i].data(), row_size, 1);
			flaten_pose.col(i) = flat_neutral_pose - Eigen::Map<Eigen::MatrixXd>(pose[i].data(), row_size, 1);
		}
	};
int get_blendshape_weight_size() const {
	return flaten_pose.cols();
	}
void blend( const Eigen::Ref<const Eigen::VectorXd> w, Eigen::MatrixXd& out) const {
	out = neutral_pose + flaten_pose * w;
};
void blend(Eigen::Ref<const Eigen::VectorXd> w, Eigen::MatrixXd& out, const std::vector<int>& lmk) const {
	out.conservativeResize(lmk.size(), 3);
	
	Eigen::Map<Eigen::MatrixXd> flat_neutral_pose( neutral_pose.data(), neutral_pose.size(), 1);
	//std::cout << w << std::endl;


	//Eigen::MatrixXd q(lmk.size(),3);
	//for (int i = 0; i < lmk.size(); i++) {
	//	q.row(i) = neutral_pose.row(lmk[i]);
	//}	
	//std::cout << "neu" << q.block(0, 0, 4, 3) << std::endl;

	//

	//for (int i = 0; i < flaten_pose.cols(); i++) {

	//Eigen::MatrixXd t = flaten_pose.col(i);
	//Eigen::Map<Eigen::MatrixXd> dt(t.data(), int(flaten_pose.col(0).size() / 3), 3);
	//Eigen::MatrixXd q(lmk.size(),3);
	//for (int i = 0; i < lmk.size(); i++) {
	//	q.row(i) = dt.row(lmk[i]);
	//}	
	//std::cout <<i << "th- mat" << std::endl;
	//std::cout << q.block(0,0,4,3) << std::endl;
	//}


	Eigen::MatrixXd mesh = neutral_pose + (flaten_pose * w);

	Eigen::Map<Eigen::MatrixXd> tmp_map(mesh.data(), neutral_pose.rows(), neutral_pose.cols());
	for (int i = 0; i < lmk.size(); i++) {
		out.row(i) = tmp_map.row(lmk[i]);
	}
	//std::cout<< out << std::endl;
};
}
;

inline void load_blendshapes(const std::string& root_pth, struct Blendshapes& bb) {
	using namespace std;

	filesystem::path root = root_pth;
	cout << root << endl;
	if (!filesystem::is_directory(root)) {
		cerr << "this path is not dir." << endl;
		return;
	}
	std::vector<string> pth_list;
	filesystem::path neutral_pth;

	for ( auto pth : filesystem::directory_iterator(root)) {
		
		if (pth.path().extension() == ".obj") {
			
			if (pth.path().string().find("identity") != std::string::npos) {
				neutral_pth = pth.path();
			}
			else {
				pth_list.push_back(pth.path().string());

			}
		}

	}
	
	sort(pth_list.begin(), pth_list.end(), compareNat);

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::cout << "neutral apth " << neutral_pth.string() << std::endl;
	
	igl::read_triangle_mesh(neutral_pth.string(), V, F);
	bb.neutral_pose = V;
	bb.F = F;

	std::vector<Eigen::MatrixXd> exprs;
	for (int i = 0; i < pth_list.size(); i++) {
		std::cout << "blendshapes : " << pth_list[i] << "was loaded." << std::endl;
		igl::read_triangle_mesh(pth_list[i], V, F);
		exprs.emplace_back(V);
	}
	bb.pose = exprs;
	bb.compile();
}

