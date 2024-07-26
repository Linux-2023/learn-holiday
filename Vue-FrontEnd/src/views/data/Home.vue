<template>
  <div style="padding: 10px">
    <el-table :data="tableData" style="width: 100%; border: none;" size="small">
      <el-table-column prop="topic" label="习题题目"></el-table-column>
      <el-table-column label="输入答案">
        <template v-slot="scope">
          <el-input v-model="scope.row.userAnswer" style="width: 100px"></el-input>
        </template>
      </el-table-column>
      <el-table-column label="确认提交">
        <template v-slot="scope">
          <el-button @click="handleJudgement(scope.row)">确认提交</el-button>
        </template>
      </el-table-column>
</el-table>


    <div style="margin: 10px 0">
      <el-pagination
          @size-change="handleSizeChange"
          @current-change="handleIndexChange"
          :current-page="pageIndex"
          :page-sizes="[5, 10, 20]"
          :page-size="pageSize"
          layout="sizes, prev, pager, next, jumper, total"
          :total="totalNum">
      </el-pagination>
    </div>
  </div>
</template>

<script>
import request from "@/api/request";

export default {
  name: 'UserData',
  components: {},

  data() {
    return {
      user: null,
      formData: {
        id: null,
        user_id: null,
        q_id: null,
        correct: null,
      }, // 增加或修改的表单数据
      tableData: [],
      totalNum: 0,
      pageIndex: 1,
      pageSize: 10
    }
  },

  created() {
    let userStr = sessionStorage.getItem("user")
    this.user = JSON.parse(userStr)
    this.load()
  },

  methods: {
    load() {
        request.get("/kt/problem", {
            params: {
                pageIndex: this.pageIndex,
                pageSize: this.pageSize,
            }
        }).then(res => {
            if (res.msg) {
                this.$message(res.msg)
                return
            }
            res.data.forEach(item => {
                item.judgement = null;
            });
            this.tableData = res.data
            this.totalNum = res.num
        })
    },

   handleJudgement(row) {
      if (row.userAnswer === row.answer) {
        this.$message.success('答案正确！');
        this.formData.correct = 1;
      } else {
        this.$message.error('答案错误！');
        this.formData.correct = 0;
      }

      this.formData.q_id = row.q_id;
      this.formData.user_id = this.user.id;

      // 发起 POST 请求
      request.post("/kt/history/add", this.formData).then(res => {
        if (res.code === 0)
          this.$message({
            type: "success", message: "成功加入历史记录"
          });
        else
          this.$message({
            type: "error", message: "加入历史记录失败"
          });
  });

  this.load();
},

    handleSizeChange(pageSize) {
        this.pageSize = pageSize
        this.load()
    },

    handleIndexChange(pageIndex) {
        this.pageIndex = pageIndex
        this.load()
    }


  }
}
</script>
