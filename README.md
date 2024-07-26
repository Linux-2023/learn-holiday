### 目录结构

- Flask-BackEnd `flask后端`
  * app `后端主体文件`
    * alg `深度学习模块`
      * data  `数据集`
      * data_process.py `数据预处理`
      * gikt.py `GIKT模型`
        * mlp.py `卷积扩散部分MLP`
        ~~* layers.py `原GraphFormers中网络层，只使用了部分，效果问题最终模型中未使用`~~
      * pebg.py `PEBG模型`
        * params.py `一些参数`
        * pnn.py `PEBG中PNN层`
      * train.py `仅模型训练`
      * train_test.py `模型训练和测试-五折交叉验证`
      * train_test2.py `模型训练和测试-4:1训练测试`
      * utils.py `工具函数`
    * view `flask蓝图`
    * \__init__.py `初始化`
    * create_data `创建初始数据`
    * entity `实体类`
    * setup `启动`
  * migrate `数据库迁移文件`

* Vue-FrontEnd `vue前端`
  * public `共用文件`
  * src `源代码`
    * api `全局请求设置`
    * assets `静态组件`
    * components `自定义vue组件`
    * layout `页面布局`
    * router `路由`
    * store `信息储存`
    * views `页面`
    * App.vue `开始文件`
    * main.js `js包引入`
  * 其他的是一些配置

### 启动

上面未提及的一些目录都在`.gitignore`，请手动添加后再启动

**前端**

进入目录`Vue-FrontEnd`

```bash
cd Vue-FrontEnd
```

安装需要的包

```bash
cnpm install
```

启动

```bash
npm run serve
```

**后端**

1. 用pycharm打开目录`Flask-BackEnd`

2. 修改mysql数据库配置项

3. 运行`data_process.py`，生成预训练数据

4. 运行`pre_train.py`，生成预训练问题向量

5. 运行`train.py`，训练并保存一次模型（以便后端调用）

6. 安装需要的包，其中：

   - pytorch==**2.1.0**
   * flask==**2.2.5**
   * Cuda==**12.1**

7. 解决报错后，运行**一次**`create_data.py`（或者在`__init__.py`的app_context中调用**一次**create_data函数），在数据库中添加初始数据

8. 重新启动，访问本机5001端口，测试系统

### 项目存在的一些问题

**算法**

- 使用GraphFormers中的中心性编码进行优化，以入读出度训练节点重要性，没有得到期望优化结果
- 参数仍可调优，最终结果可以继续优化

**后端**

- 包引用（尤其是对算法包`alg`的引用）存在问题，使用了粗暴的解决方式 `sys.path.append()` ，且无法使用相对路径导入
- flask数据库迁移会报错，只能自己手动通过DBMS修改



