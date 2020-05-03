# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recommendation generation module."""

import logging
import numpy as np
import os
import pandas as pd

import google.auth
import google.cloud.storage as storage

logging.basicConfig(level=logging.INFO)

LOCAL_MODEL_PATH = '/tmp'

ROW_MODEL_FILE = 'model/row.npy'
COL_MODEL_FILE = 'model/col.npy'
USER_MODEL_FILE = 'model/user.npy'
ITEM_MODEL_FILE = 'model/item.npy'
# save the model
USER_ITEM_DATA_FILE = 'ml-data/MSE_OAE_Cat_ML.csv'


class Recommendations(object):
  """Provide recommendations from a pre-trained collaborative filtering model.

  Args:
    local_model_path: (string) local path to model files
  """

  def __init__(self, local_model_path=LOCAL_MODEL_PATH):
    _, project_id = google.auth.default()
    self._bucket = 'recserve_' + project_id
    self._load_model(local_model_path)

  def _helper_make_identifier(self,df):
    str_id = df.apply(lambda x: '_'.join(map(str, x)), axis=1)
    return pd.factorize(str_id)[0]

  def _pre_process(self,view_df):
    """ Generate wound_ID
    :param view_df:
    :return: modified data files
    """
    view_df.columns = ["Assignment_ID", "Logged_In", "User_ID", "Preformed",
                           "Wound_duration", "Type_of_wound", "Wound_healing_phase",
                           "Necrotic", "Infection", "Moisture", "Survival", "Painful_experience",
                           "Edge_Callus", "Edge_Cyanotic", "Edge_Inverted", "Edge_Inert", "Edge_Maceration",
                           "Edge_Normal", "Edge_Reepithelizing", "Skin_type", "Compression", "Filling", "Covering",
                           "Absorbing", "Fixing",
                           "Cleaning", "Reducing_pressure", "Debridement", "Combating_infection", "Restore_biobalance",
                           "Article"]

    df_shopping = view_df.drop(['Assignment_ID'], axis=1)

    # make_identifier already o-based

    df_shopping['Wound_ID'] = self._helper_make_identifier(
      df_shopping[['Preformed', 'Wound_duration', 'Type_of_wound', 'Wound_healing_phase',
                   'Necrotic', 'Infection', 'Moisture', 'Survival', 'Painful_experience',
                   'Edge_Callus', 'Edge_Cyanotic', 'Edge_Inverted', 'Edge_Maceration',
                   'Edge_Inert', 'Edge_Normal', 'Edge_Reepithelizing', 'Skin_type', 'Compression',
                   'Filling', 'Covering', 'Absorbing', 'Fixing', 'Cleaning', 'Reducing_pressure',
                   'Debridement', 'Combating_infection', 'Restore_biobalance']])

    ratings_df = df_shopping[['Article', 'Wound_ID']]
    ratings_df = ratings_df.groupby(["Article", "Wound_ID"]).size().reset_index(name="Times")
    ratings_df['article_id_rearranged'] = self._helper_make_identifier(ratings_df[['Article']])
    ratings_df.columns = ['article_id', 'wound_id', 'rating', 'article_id_rearranged']
    ratings_df = ratings_df[['wound_id', 'article_id_rearranged', 'rating', 'article_id']]

    ratings_df[ratings_df.columns] = ratings_df[ratings_df.columns].apply(pd.to_numeric, errors='coerce')


    return ratings_df
  def _load_model(self, local_model_path):
    """Load recommendation model files from GCS.

    Args:
      local_model_path: (string) local path to model files
    """
    # download files from GCS to local storage
    os.makedirs(os.path.join(local_model_path, 'model'), exist_ok=True)
    os.makedirs(os.path.join(local_model_path, 'ml-data'), exist_ok=True)
    client = storage.Client()
    bucket = client.get_bucket(self._bucket)

    logging.info('Downloading blobs.')

    model_files = [ROW_MODEL_FILE, COL_MODEL_FILE, USER_MODEL_FILE,
                   ITEM_MODEL_FILE, USER_ITEM_DATA_FILE]
    for model_file in model_files:
      blob = bucket.blob(model_file)
      with open(os.path.join(local_model_path, model_file), 'wb') as file_obj:
        blob.download_to_file(file_obj)

    logging.info('Finished downloading blobs.')

    # load npy arrays for user/item factors and user/item maps
    self.user_factor = np.load(os.path.join(local_model_path, ROW_MODEL_FILE))
    self.item_factor = np.load(os.path.join(local_model_path, COL_MODEL_FILE))
    self.user_map = np.load(os.path.join(local_model_path, USER_MODEL_FILE))
    self.item_map = np.load(os.path.join(local_model_path, ITEM_MODEL_FILE))

    logging.info('Finished loading arrays.')

    # load user_item history into pandas dataframe
    views_df = pd.read_csv(os.path.join(local_model_path,
                                        USER_ITEM_DATA_FILE), sep=',', header=0)
    views_df = self._pre_process(views_df)
    self.user_items = views_df.groupby('wound_id')

    logging.info('Finished loading model.')

  def get_recommendations(self, user_id, num_recs):
    """Given a user id, return list of num_recs recommended item ids.

    Args:
      user_id: (string) The user id
      num_recs: (int) The number of recommended items to return

    Returns:
      [item_id_0, item_id_1, ... item_id_k-1]: The list of k recommended items,
        if user id is found.
      None: The user id was not found.
    """
    article_recommendations = None

    if user_id in self.user_map:
      recommendations = generate_recommendations(user_id,
                                                 self.user_factor,
                                                 self.item_factor,
                                                 num_recs)

      # map article indexes back to article ids
      article_recommendations = [self.item_map[i] for i in recommendations]

    return article_recommendations


def generate_recommendations(user_idx, row_factor, col_factor, k):
  """Generate recommendations for a user.

  Args:
    user_idx: the row index of the user in the ratings matrix,
    row_factor: the row factors of the recommendation model

    col_factor: the column factors of the recommendation model

    k: number of recommendations requested

  Returns:
    list of k item indexes with the predicted highest rating,
    excluding those that the user has already rated
  """

  # bounds checking for args
  # assert (row_factor.shape[0] - len(user_rated)) >= k

  # retrieve user factor
  user_f = row_factor[user_idx]

  # dot product of item factors with user factor gives predicted ratings
  pred_ratings = col_factor.dot(user_f)

  # find candidate recommended item indexes sorted by predicted rating
  k_r = k
  candidate_items = np.argsort(pred_ratings)[-k_r:]

  recommended_items = [i for i in candidate_items]
  recommended_items = recommended_items[-k:]

  # flip to sort highest rated first
  recommended_items.reverse()

  return recommended_items

