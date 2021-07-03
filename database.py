from sqlalchemy import create_engine,MetaData,Table,String,Column, Integer,DATE

def create_db():

    engine=create_engine('sqlite:///replays.db')
    meta=MetaData()

    replays=Table('Replays',meta, Column('id', Integer, primary_key = True),
       Column('moves', String),
       Column('date', String))
    meta.create_all(engine)

