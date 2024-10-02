# -*- coding: utf-8 -*-
"""
Created on: 9/11/2024
Original author: Adil Zaheer
"""
import multiprocessing
import time
from src.nts_processing.destination_choice_ml_modelling.model_functions import \
    transform_data_pre_modelling, feature_selection, load_model, train_trip_rate_model, cost_func, \
    prep_data_gravity, grav_model
from src.nts_processing.processing_classified_build.process_cb_functions import \
    process_cb_destination_choice


# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position


def main(params):
    n_cores = multiprocessing.cpu_count()
    print(f"Using {n_cores} CPU cores")


    if (params.output_folder / 'regression_method.pkl').exists():
        print('Loading saved model, hyperparameters and data for modelling')
        model, hyperparameters, final_data_to_model = load_model(output_folder=params.output_folder)

        df_grav_prepared = prep_data_gravity(output_folder=params.output_folder,
                                             ppm_ppk_data=params.ppm_ppk_data)

        data_w_cost_func = cost_func(data=df_grav_prepared)

        grav_model_results = grav_model(gen_cost_data=data_w_cost_func,
                                        x_pred=final_data_to_model,
                                        target_column=params.target_column,
                                        model=model,
                                        hyperparameters=hyperparameters)



        return


    else:
        start_time = time.time()

        initial_data = process_cb_destination_choice(data=params.data,
                                                     output_folder=params.output_folder,
                                                     columns_to_model=params.columns_to_model,
                                                     purpose_value=params.purpose_value)

        final_data_to_model = transform_data_pre_modelling(data=initial_data,
                                                           output_folder=params.output_folder,
                                                           index_columns=params.index_columns,
                                                           drop_columns=params.drop_columns,
                                                           numerical_features=params.numerical_features,
                                                           categorical_features=params.categorical_features,
                                                           target_column=params.target_column)

        trip_rate_model = train_trip_rate_model(model_type=params.model_type,
                                                data=final_data_to_model,
                                                target_column=params.target_column,
                                                n_cores=params.n_cores,
                                                output_folder=params.output_folder)

        trips_model, trips_hyperparameters = trip_rate_model.run()


        df_grav_prepared = prep_data_gravity(output_folder=params.output_folder,
                                             ppm_ppk_data=params.ppm_ppk_data)
        data_w_cost_func = cost_func(data=df_grav_prepared)
        grav_model_results = grav_model(gen_cost_data=data_w_cost_func,
                                        x_pred=final_data_to_model,
                                        target_column=params.target_column,
                                        model=trips_model,
                                        hyperparameters=trips_hyperparameters)



        end_time = time.time()
        print(f"Total run time: {end_time - start_time:.2f} seconds")

    return



'''

    # Predict trip rates using the trained model
    X_pred = df_grav_prepared.drop(columns=[params.target_column])
    predicted_trip_rates = model.predict(X_pred)

    # Calculate trips using the gravity model
    trips = calculate_trips(predicted_trip_rates, gen_cost, 
                            df_grav_prepared['pop_o'], df_grav_prepared['emp_d'])

'''









# input zone system where we have a cost matrix between zones, got land use data so who lives where etc, and emp land use so we
# know what jobs are in each area/ attractions etc. train model on nts where you know where people make the trisp
# prediction is we use that model on our lu, cost, zone system etc to build a matrix
