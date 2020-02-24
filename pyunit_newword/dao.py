from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text, create_engine, func)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DB_CONNECT = 'mysql+pymysql://root:mysql@127.0.0.1:3306/guiyang12345?charset=utf8'
engine = create_engine(DB_CONNECT, echo=False, encoding='utf-8')
Session = sessionmaker(bind=engine)
session = Session()
BASE = declarative_base(engine)


class Data(BASE):
    __tablename__ = 'nls_asr_short_log'
    id = Column(String(255), primary_key=True, autoincrement=True)
    url = Column(String(255))
    voice_name = Column(String(255))
    original_text = Column(Text)
    recognize_text = Column(Text)
    sample_rate = Column(String(255))
    create_time = Column(DateTime)


