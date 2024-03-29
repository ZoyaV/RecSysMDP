TIMESTAMP_COL = 'timestamp'
USER_ID_COL = 'user_id'
ITEM_ID_COL = 'item_id'
RELEVANCE_CONT_COL = 'relevance_cont'
RELEVANCE_INT_COL = 'relevance_int'
RATING_COL = 'rating'
REWARD_COL = 'reward'
TERMINATE_COL = 'terminate'


def relevance_col_for_rating(discrete: bool):
    return RELEVANCE_INT_COL if discrete else RELEVANCE_CONT_COL
