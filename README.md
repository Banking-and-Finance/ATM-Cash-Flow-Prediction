# ATM-Cash-Flow-Prediction

## Abstract
Physical cash is the most liquid asset and holds high usage rates despite rapid technological advancements in the digital cash industry, especially in the context of a developing country like India. As such, Automated Teller Machines (ATMs) form an integral component of any individual's cash needs and managing the amount of cash in any given ATM is of paramount  importance for any bank. ATMs are periodically filled with cash by banks, predominantly with the help of cash management agencies. Through our research we want to determine the feasibility of adding mathematical support to the current system by using Data Analysis and Machine Learning (ML). The findings of this research and modelling approaches were applied to a procured dataset which consists of data of 5 ATMs of the same bank from 2011-2017. The year, month, day, festival religion, day of the week, holiday status of yesterday, today and tomorrow were the factors that were initially analyzed and later on, average withdrawal for a given window of time in the past, the weather status and the type of holiday were the other features engineered which proved to be useful. The algorithms being tested include Lasso, Ridge, Elasticnet, Random Forest & CatBoost which is a deep learning framework. Two approaches for testing were determined, one approach was to create one model for all years in the data and another approach was to develop a model per year. In both approaches, it was found that Lasso regression outperformed all the other algorithms in contention.

## Introduction
India has a physical cash-based economy and despite the continuous technological advancements in the field of digital cash-based systems such as Unified Payments Interface (UPI), Wallets, etc. having currency notes is vital for most situations [1]. The amount of currency in circulation in the financial year 2020 was ₹26.74 trillion, 22.7% higher than the previous year, according to the Reserve Bank of India's (RBI) annual report [2]. \
It follows that managing cash logistics is important for many segments such as retail, merchant services, agricultural sector, commercial banks, etc.
According to RBI’s January 2020 statistics, there are in total 2,10,263 registered Automated Teller Machines (ATM) installed all over India both on and off-site (1,12,944 & 97,319). [3] \
Cash withdrawals, deposits, transfer funds and obtaining account information are common functions of ATMs. \
Managing the amount of currency in ATMs is of utmost importance to banks, as undershooting the demand can lead to stock-out which leaves customers highly frustrated and overshooting the demand creates dead cash which could have been distributed to other ATMs/branches or services that required it. This creates the scope for adding mathematical support to the process of determining the amount to be refilled in ATMs or ATM Cash Demand Prediction, aided by techniques such as Machine Learning, Neural Networks, etc. \
From multiple banks that were surveyed in Mumbai, India, it was found that a majority of the banks lease out their cash management duties to third-party agencies although there were a couple that did not use such facilities. On a broader scale, the cash management industry is moderately concentrated in India and is projected to grow even further as government schemes supporting more ATMs and ATM facilities for rural areas increases [1]. \
The current cash management cycle involves cash pick up from bank, cash movement from the bank to the ATMs' locations, grading, counting, monitoring ATMs for assessing the level of cash in an ATM and replenishing cash in the machine accordingly. [1] \
This research aims to determine the factors involved with predicting the cash demand for an ATM and try to explore the possibilities of an in-house cash demand prediction system for a particular bank. 

## Domain Research

### Literature Survey
There have been many researches tackling various aspects of ATM Management such as determining optimal ATM location, optimizing route for cash replenishment in ATMs, forecasting the cash demand for ATMs and some researches have even attempted a combination of two or more of these problems as summarized below. \
S. Madhavi et al. (2014) [4] provided a high-level overview of the potential of using data mining & data analysis on ATM transaction data in order to determine peak hours as well popular ATM services used by customers which in turn the bank can use to provide better service to them. \
M. Genevois et al. (2015) [5] looked at ATM Location as well as Cash Management as two parts of a larger problem. They discussed the various criteria that are involved in deciding the location site for an ATM such as foot traffic per hour, drive-time traffic, heaviest traffic locations, physical security, proximity of an ATM within a certain radius, type of neighbourhood, presence of hotels, etc. with an exhaustive literature survey on both problems. \
R. Simutis et al. (2008) [6] used simulated as well as real data from 15 ATMs in order to compare a flexible ANN model versus a Support Vector Regression (SVR) model and used Mean Absolute Percentage Error (MAPE) as the metric to conclude that ANNs performed better than SVRs. They used their ANN model schema to improve the performance capabilities of a professional cash management software called ASOMIS. \
R. Armenise et al. (2010) [7] proposed a Genetic Algorithm (GA) approach to solve the ATM refilling problem using data from 30 Italian ATMs and have found the GA approach to be feasible but stated the potential of further optimization and handling non-stationarity of data. \
S. Teddy et al. (2010) [8] used a novel approach termed as Pseudo Self-Evolving Cerebellar Model Articulation Controller (PSECMAC) on the NN5 ATM competition dataset which involves 111 time-series. Their research claimed that PSECMAC fared better when compared to other global-learning based benchmarks and that use of local learning-based Computational Intelligence (CI) models are promising alternatives to the commonly used global learning-based approaches for time series modelling and prediction. They achieved this by using just the time series data with no account of seasonality or any other factor which leaves the potential for further research in this domain. \
A. Brentnall et al. (2010) [9] used data from 190 ATMs based in the United Kingdom spread over 2 years. They utilized different models such as linear models, autoregressive models, structural time series models and Markov-switching models and compared them. \
Y. Ekinci et al. (2015) [10] looked at the prospects of grouping ATMs into clusters and providing a group forecast before breaking it down into individual forecasts for each ATM. The grouping is done based on the nearest neighbourhood approach as it was observed that ATMs that are nearby show similar trends in withdrawal behaviour. It is also stated that providing group forecasts would reduce transportation costs to refill the ATMs, as all ATMs in the same group would be close to each other, thereby reducing overall travelling cost. They demonstrated their results for a single cluster of 5 ATMs in which 3 of them showed more accurate results (a metric used was MAPE) in the group forecast approach rather than making individual forecasts which on average made the group forecast approach more accurate. \
K. Perera et al. (2018) [11] conducted surveys in the Colombo district of Sri Lanka, where they surveyed bank officials to understand the current scenario of the cash demand forecasting systems and what are the important factors according to them and using this data they surveyed customers to conclude on important factors that affect ATM cash withdrawal. According to their research, location of the ATM, holidays and its impact, salary days and bonus days of the customers, day of the week and weather condition of the nearby area were factors that affected cash withdrawal demand. They used real data from a single ATM and two approaches for modelling, regression and time series out of which time series performed the best. 

### Discussions with Bank officials
To understand current industry standards and the practical knowledge behind the working of an actual ATM refilling system, meetings were held with officials from different banks. The summarization of the discussions with bank officials are as follows: \ 
- Most banks do not have any ATM cash management system but resort to cash management agencies for prediction and refilling wherein the bank only supplies cash as asked by the agency weekly. The agency has a vaulting system where they stock the excess cash if it isn’t utilised within that week and carried forward to the next week. So the banks have no control over the prediction.
- Banks that can’t afford cash management agencies often make approximate predictions and calculations introspectively based on past experiences.
- ATMs next to bank branches are filled by the respective bank itself.
- Some banks take reference of parameters used in forecasting for determining new ATM sites.
- Some parameters that could help the model for generalizing better are as follows,
    - A. Did the ATM run out of cash and the time at which it ran out.
    - B. Cash denominations inserted as well as withdrawn from ATM.
    - C. ATM location data like average income, population, etc. 
    - D. Number of failed transactions for that ATM.
    - E. Information about when the ATM was closed due to maintenance or failure. 

## Data Analysis



