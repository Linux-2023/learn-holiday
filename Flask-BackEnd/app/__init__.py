"""
初始化文件
"""
import os.path

from flask import Flask
from flask_migrate import Migrate
from flask_cors import CORS
import sys
from sql_init import db

sys.path.append(os.path.join(os.path.dirname(__file__), 'alg'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'view'))
sys.path.append('C:\KT\Flask-BackEnd')
# 解决了算法模块无法引用的问题
sys.path.append('C:\.conda\envs\JY\Lib\site-packages/flask')
# 解决了数据库迁移找不到包的问题

# 主机名
HOSTNAME = '127.0.0.1'
# mysql端口号
PORT = 3306
# 用户名
USERNAME = 'root'
# 密码
PASSWORD = '1234'
# 数据库名称
DATABASE = 'mysql'


# 主文件配置


def create_app(app):
    # 声明

    migrate = Migrate(app, db)

    # 这时候才导入蓝图，可以避免循环引用
    from app.view.user_bp import user_bp
    from app.view.skill_bp import skill_bp
    from app.view.kt_bp import kt_bp
    from app.view.question_bp import question_bp

    app.register_blueprint(user_bp)
    app.register_blueprint(skill_bp)
    app.register_blueprint(kt_bp)
    app.register_blueprint(question_bp)

    # 测试内容
    with app.app_context():
        # 生成初始数据，只需跑一次即可
        # from app.create_data import create_data
        # create_data()
        pass

    # ic(app.config) # 打印app配置


if __name__ == '__main__':
    app = Flask(__name__, instance_relative_config=True)
    cors = CORS(app)

    # 设置数据库配置,连接格式：dialect://username:password@host:port/database
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}'
    # mysql8版本以上不需要加任何参数
    app.config['DEBUG'] = True
    app.config['ENV'] = 'development'
    app.config['FLASK_APP'] = 'app'
    db.init_app(app)
    create_app(app)
    app.run()
