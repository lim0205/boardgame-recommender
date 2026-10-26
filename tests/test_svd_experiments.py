import json
import pandas as pd
from boardrec.demo_data import generate_demo
from boardrec.preprocessing import PrepConfig, prepare
from boardrec.svd_experiments import run_svd_experiments


def test_validation_only_and_artifacts(tmp_path):
    raw = generate_demo(tmp_path/'raw')
    prepared = tmp_path/'prepared'
    prepare(raw/'user_ratings.csv', raw/'games.csv', raw/'mechanics.csv',
            raw/'themes.csv', prepared,
            PrepConfig(top_games=60,min_user_ratings=8,max_users=12,seed=42))
    first = run_svd_experiments(prepared, tmp_path/'results1',
                                factors=(2,3),scales=(0.5,1.0),max_hit_users=5)
    assert first['best_hyperparameters']['n_factors'] in (2,3)
    assert 1 <= first['validation_metrics']['rmse'] <= 10
    for name in ['validation_metrics.csv', 'results.json', 'validation_rmse.svg']:
        assert (tmp_path/'results1'/name).stat().st_size > 0
    # Evaluation must not depend on TEST data at all.
    test_path = prepared/'test.csv'
    test_df = pd.read_csv(test_path)
    test_df['rating'] = 1.0
    test_df.to_csv(test_path,index=False)
    second = run_svd_experiments(prepared, tmp_path/'results2',
                                 factors=(2,3),scales=(0.5,1.0),max_hit_users=5)
    assert first == second
