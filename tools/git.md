git

一 创建git仓库

1. 创建SSH Key

   ```
   $ ssh-keygen -t rsa -C "youremail@example.com"
   ```

   在.ssh目录下有id_rsa和id_rsa.pub两个文件。

2. 在github上将id_rsa.pub公钥添加到SSH Keys中

3. 在本地上创建一个git仓库

   ```
   $ git init
   ```

4. 关联远程库

   ```
   $ git remote add origin git@github.com:zhongshanhao/learngit.git
   ```

   origin是远程仓库的名字，关联远程库后就可以在本地管理git仓库了，这里的仓库是learngit。
   
   

二 向开源社区贡献自己的代码

1. 在github上fork项目源代码到自己的远程仓库

2. 在本地clone仓库

   ```
   git clone git@github.com:zhongshanhao/tidb.git
   ```

3. 添加远程仓库分支upstream，该分支是项目源代码所在的地方，不是自己克隆的远程仓库

   ```
   git remote add upstream https://github.com/pingcap/tidb
   ```

4. 在本地仓库创建分支进行开发

   ```
   git checkout -b my_branch
   ```

5. 完成代码开发后，将本地master分支更新到，与upstream/master分支同步

   ```
   git fetch upstream
   git checkout master
   git merge upstream/master
   ```

   命令拓展

   ```
   git log upstream/master    # 查看upstream/master分支最新的commit
   
   # 当当前工作没有提交到本地仓库时，可以将工作现场存储起来
   git stash
   git stash list 				# 查看暂存区列表
   git stash pop			 # 恢复工作现场，删除暂存区内容
   git stash apply stash@{0}   # 恢复指定暂存区，且不删除该暂存区内容
   git stash drop 			 # 删除暂存区内容
   ```

6. 将my_branch分支变基，即将该分支的分叉出改为最新的master头结点处

   ```
   git checkout my_branch
   git rebase master
   git push origin my_branch:my_branch
   ```

7. 在github上选择my_branch分支，点击new pull request创建新的PR



图解

<img src="http://kmknkk.oss-cn-beijing.aliyuncs.com/image/git.jpg" alt="">



<img src="https://img-blog.csdnimg.cn/20190311173112758.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ppb2hvX2NoZW4=,size_16,color_FFFFFF,t_70">