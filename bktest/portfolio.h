#pragma once
#pragma once
#include "data_frame.h"
#include <cassert>
#include <type_traits>
#include <numeric>
//#include <cassert>
template <typename T>
class portfolio {
private:
	data_frame<T> weights_df;
	double cash;
	data_frame<T> shares_df;
	data_frame<T> gross_exposure_df;
	data_frame<T> trade_price_df;
	double liquidity; // value of cash + long position + proceeds from short position
	double commission_rate_dbl;
	double proceeds;
	double tade_exposure_limit; // total exposure limit available for each trade, a percentage double
public:
	portfolio() {};
	portfolio(data_frame<T> weights, double cash);
	~portfolio() {};
	data_frame<T> get_weights();
	double get_cash();
	void set_cash(double _cash);
	void set_weights(data_frame<T> _weights);
	void run_test(data_frame<T> price_df);
	void liquidity_cal();
	double trade(Eigen::MatrixXd prices, Eigen::MatrixXd& shares, bool simu);//trade simulation to test whether the trade would be successfully executed
	void trade(Eigen::MatrixXd prices, Eigen::MatrixXd shares,T start_date);//actual trade
};


template <typename T>
portfolio<T>::portfolio(data_frame<T> weights, double _cash) {
	this->cash = _cash;
	this->weights_df = weights;
	this->liquidity = _cash;
}

template <typename T>
data_frame<T> portfolio<T>::get_weights() {
	return this->weights_df;
}

template <typename T>
double portfolio<T>::get_cash() {
	return this->cash;
}

template <typename T>
void portfolio<T>::set_cash(double _cash) {
	this->cash = _cash;
}

template <typename T>
void portfolio<T>::set_weights(data_frame<T> _weights) {
	this->weights_df = _weights;
}


template <typename T>
void portfolio<T>::run_test(data_frame<T> price_df) {
	static_assert(std::is_same<T, boost::posix_time::ptime>::value, "price_df must be a data_frame of type boost::posix_time::ptime for combine price and portfolio weights");
	std::vector<T> weight_index = this->weights_df.get_index();
	std::vector<T> price_index = price_df.get_index();
	std::vector<double> ror;
	int beg = 0;
	int end = 0;
	T start_date, end_date;
	int i;
	for (i = 0; i < weights_df.rows(); i++) {
		start_date = weight_index[i];

		if (i < weights_df.rows() - 1) {
			end_date = weight_index[i + 1];
		}
		else {
			boost::gregorian::days daysadded(1);

			end_date = price_index[price_index.size() - 1] + daysadded;
		}

		while (end < (price_index.size() - 1) and price_index[end] < end_date) {

			if (price_index[end] == start_date) {
				beg = end;
			}

			end++;

		}
		//order execute
		std::vector<int> row_idx(end - beg + 2);
		std::iota(row_idx.begin(), row_idx.end(), beg - 1);
		data_frame<T> price_df_slice = price_df.get_rows(row_idx);
		data_frame<T> ret_df = price_df_slice.log().row_diff().dropna();
		//get new weight
		Eigen::MatrixXd weight_mx = this->weights_df.get_rows(i).get_data();
		Eigen::MatrixXd last_weight_mx;
		//if in the first period ,set the last weights as a zero vector
		if (i == 0) {
			last_weight_mx = Eigen::MatrixXd::Zero(1, weight_mx.cols());
		}
		else {
			last_weight_mx = this->weights_df.get_data().row(i - 1);
		}


		//compute current pfl's total value = cash(include proceeds from short selling) + positive asset_exposure
		liquidity_cal();
		//update aseet val for each assets by change of shares compared with last period
		Eigen::MatrixXd sub_price_mx = price_df_slice.get_data().row(0);
		Eigen::MatrixXd weight_chg_mx = weight_mx - last_weight_mx;
		Eigen::MatrixXd share_chg_mx = Eigen::MatrixXd::Zero(1, weight_mx.cols());
		double liquid_needed_dbl = 0; // the money you will pay after buying + selling plus transaction cost and borrow fee and other fees TBD
		double long_share_dbl = 0;
		double short_share_dbl = 0;
		double tmp_proceeds = 0;
		double proceeds_chg = 0;
		// need to recalculate the liquid value after each trade, in above line
		// calcualte available exposure for each trade, considering the trade limit, which is set to guarantee that each target position is set successfully
		
		// suppose we employ market netural strategy by assign same dollor exposure to each positon
		

		//suppose we trade at sub_price_mx price, may need further clarification on trading price
		Eigen::MatrixXd trade_price_mx = sub_price_mx;
		double liquid_needed = trade(trade_price_mx, weight_chg_mx, true);
		//check condition
		if (liquid_needed_dbl > this->liquidity) {
			std::cout << "Not enough cash to trade" << std::endl;
			return;
		}
		else {
			//the target portfolio can be implemented
			trade(trade_price_mx, weight_chg_mx,start_date);
		}
		Eigen::MatrixXd new_share_mx = this->shares_df.get_rows({ this->shares_df.rows() - 1 }).get_data();
		//daily return compute
		Eigen::MatrixXd value_mx = new_share_mx.cwiseProduct(trade_price_mx);
		value_mx = value_mx * ret_df.get_data();
		data_frame<T> sub_value_df(value_mx, ret_df.get_column_names(), ret_df.get_index());
		gross_exposure_df.row_concat(sub_value_df);
	}
	std::cout << "Gross Exposure:" << std::endl;
	gross_exposure_df.head(gross_exposure_df.rows());
}


template <typename T>
double portfolio<T>::trade(Eigen::MatrixXd prices, Eigen::MatrixXd& shares, bool simu) {
	double liquid_val_off_fee_dbl = liquidity * (1 - commission_rate_dbl) * tade_exposure_limit;
	double liquid_needed = 0; // the money you will pay after buying + selling plus transaction cost and borrow fee and other fees TBD
	double short_exposure = liquid_val_off_fee_dbl;
	//suppose we trade at sub_price_mx price, may need further clarification on trading price
	for (int i = 0; i < shares.cols(); i++) {
		if (shares(0, i) > 0) {
			//buy
			shares(0, i) = std::floor(shares(0, i) * liquid_val_off_fee_dbl / prices(0, i));
			liquid_needed += shares(0, i) * prices(0, i) * (1 + commission_rate_dbl);
		}
		else{
			//sell
			shares(0, i) = std::floor(shares(0, i) * short_exposure / prices(0, i));
			liquid_needed -= -shares(0, i) * prices(0, i) * (1 - commission_rate_dbl);; // the money you will get after selling - transaction cost and borrow fee
		}
	}
	return liquid_needed;
}

template <typename T>
void portfolio<T>::trade(Eigen::MatrixXd prices, Eigen::MatrixXd shares,T start_date) {
	//suppose we trade at sub_price_mx price, may need further clarification on trading price
	Eigen::MatrixXd last_shares = this->shares_df.get_rows({ this->shares_df.rows() - 1 }).get_data();
	double proceed = 0.0;
	for (int i = 0; i < shares.cols(); i++) {
		if (shares(0, i) > 0) {
			//buy or cover
			//cover situation
			if (last_shares(0, i) < 0){
				this->proceeds -= std::min(shares(0, i), -last_shares(0, i)) * trade_price_df(trade_price_df.rows() - 1, i);
				//if added shares are more than the short shares, then the rest will be bought
				this->cash -= std::max(shares(0, i) + last_shares(0, i), 0.0) * prices(0, i) * (1 + commission_rate_dbl);
				//possibly need to deduct the interest incured by shorting
				// TBD
			}
			else {
				this->cash -= shares(0, i) * prices(0, i) * (1 + commission_rate_dbl);
			}

		}else{
			//sell
			proceed = -shares(0, i) * prices(0, i) * (1 - commission_rate_dbl);
			this->cash += proceed;// the money you will get after selling - transaction cost and borrow fee
			this->proceeds += proceed;
		}
	}
	//add new shares
	Eigen::MatrixXd new_share_mx = shares + this->shares_df.get_rows({ this->shares_df.rows() - 1 }).get_data();
	data_frame<T> new_share_df(new_share_mx, this->shares_df.get_column_names(), std::vector<T> {start_date});
	this->shares_df.row_concat(new_share_df);
	//add new price
	data_frame<T> new_price_df(prices, trade_price_df.get_column_names(), std::vector<T> {start_date});
	this->trade_price_df.row_concat(new_price_df);
}
template <typename T >
void portfolio<T>::liquidity_cal() {
	//calculate the liquidity of the portfolio
	Eigen::MatrixXd last_exp = gross_exposure_df.get_rows({ gross_exposure_df.rows() - 1 }).get_data();
	double pos_exp = 0.0;
	for (int i = 0; i < last_exp.cols();i++) {
		pos_exp += std::max(last_exp(0,i), 0.0);
	}
	this->liquidity = pos_exp + this->cash;
}