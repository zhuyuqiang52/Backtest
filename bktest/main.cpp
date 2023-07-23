#include "portfolio.h"

int main() {
	//alb
	data_frame<std::string> alb = data_frame<std::string>::read_csv("e:\\study\\22fall\\hf\\data\\hw1\\alb.csv", 0).get_columns({ "close" });
	alb.head(10);
	//goog
	data_frame<std::string> goog = data_frame<std::string>::read_csv("e:\\study\\22fall\\hf\\data\\hw1\\goog.csv", 0).get_columns({ "close" });
	//goog.head(10);
	data_frame<std::string> nio = data_frame<std::string>::read_csv("e:\\study\\22fall\\hf\\data\\hw1\\nio.csv", 0).get_columns({ "close" });
	//nio.head(10);
	data_frame<std::string> xom = data_frame<std::string>::read_csv("e:\\study\\22fall\\hf\\data\\hw1\\xom.csv", 0).get_columns({ "close" });
	data_frame<std::string> cmb = goog.left_join(alb);
	cmb = cmb.left_join(nio);
	cmb = cmb.left_join(xom);
	//std::cout<<"\nna removed"<<std::endl;
	data_frame<std::string> cmb_rmna_df = cmb.dropna(0, "any");
	cmb_rmna_df.set_column_names({ "goog","alb","nio","xom" });
	//cmb_rmna_df.head(300);
	//portfolio setting
	Eigen::MatrixXd weight_mx(4, 4);
	weight_mx << 0.25, 0.25, 0.25, 0.25,
		0.35, 0.15, 0.25, 0.25,
		0.15, 0.25, 0.35, 0.25,
		0.35, 0.35, 0.15, 0.15;
	data_frame<std::string> weight_df(weight_mx, { "goog","alb","nio","xom" }, { "2021-09-14","2021-12-20","2022-03-01","2022-06-01" });
	weight_df.head(10);

	//trasform index
	std::vector<boost::posix_time::ptime> ptime_idx;
	std::vector<std::string> str_index = weight_df.get_index();
	for (size_t i = 0; i < str_index.size(); i++) {
		ptime_idx.push_back(data_frame<std::string>::string_to_ptime(str_index[i]));
	}
	data_frame<boost::posix_time::ptime> weight_ptdf = data_frame<std::string>::set_index(weight_df, ptime_idx);
	//price
	std::vector<boost::posix_time::ptime> ptime2_idx;
	std::vector<std::string> str2_index = cmb_rmna_df.get_index();
	for (size_t i = 0; i < str2_index.size(); i++) {
		ptime2_idx.push_back(data_frame<std::string>::string_to_ptime(str2_index[i]));
	}
	data_frame<boost::posix_time::ptime> price_ptdf = data_frame<std::string>::set_index(cmb_rmna_df, ptime2_idx);
	//portfolio
	portfolio<boost::posix_time::ptime> pfl(weight_ptdf, 0.0);
	pfl.run_test(price_ptdf);
	return 0;
}