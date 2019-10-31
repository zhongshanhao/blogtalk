git

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

