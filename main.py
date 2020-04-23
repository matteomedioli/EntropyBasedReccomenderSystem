from recommender_system import RS

rs = RS("data/rel.rating")
rs.info()
target = 420
rs.set_target_user(target)
topN = rs.topN(50, 0.4, 0.56, cos_vector_type='chen', filtering='count', verbose=False)
