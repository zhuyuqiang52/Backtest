#include "portfolio.h"

int main() {
	//ALB
	data_frame<std::string> ALB = data_frame<std::string>::read_csv("E:\\study\\22fall\\hf\\data\\hw1\\ALB.csv", 0).get_columns({ "Close" });

	ALB.head(10);
	//GOOG
	data_frame<std::string> GOOG = data_frame<std::string>::read_csv("E:\\study\\22fall\\hf\\data\\hw1\\GOOG.csv", 0).get_columns({ "Close" });
	//GOOG.head(10);
	data_frame<std::string> NIO = data_frame<std::string>::read_csv("E:\\study\\22fall\\hf\\data\\hw1\\NIO.csv", 0).get_columns({ "Close" });
	//NIO.head(10);
	data_frame<std::string> XOM = data_frame<std::string>::read_csv("E:\\study\\22fall\\hf\\data\\hw1\\XOM.csv", 0).get_columns({ "Close" });
	data_frame<std::string> cmb = GOOG.left_join(ALB);
	cmb = cmb.left_join(NIO);
	cmb = cmb.left_join(XOM);
	//std::cout<<"\nna removed"<<std::endl;
	data_frame<std::string> cmb_rmna_df = cmb.dropna(0, "any");
	cmb_rmna_df.set_column_names({ "GOOG","ALB","NIO","XOM" });
	//cmb_rmna_df.head(300);
	//portfolio setting
	Eigen::MatrixXd weight_mx(4, 4);
	weight_mx << 0.25, 0.25, 0.25, 0.25,
		0.35, 0.15, 0.25, 0.25,
		0.15, 0.25, 0.35, 0.25,
		0.35, 0.35, 0.15, 0.15;
	data_frame<std::string> weight_df(weight_mx, { "GOOG","ALB","NIO","XOM" }, { "2021-09-14","2021-12-20","2022-03-01","2022-06-01" });
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