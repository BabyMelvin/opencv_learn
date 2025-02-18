# 配置c_cpp_properties

设置头文件，用于编译器提示

# 配置tasks.json

设置编译器，编译命令，调试命令等。

其中label要和launch.json中的`preLaunchTask`配置对应。

# 配置launch.json

设置调试器，启动命令，调试端口等。

这个设置快捷按钮的配置。

# cwd配置不对
可能出现第一行代码，段错误。


# 几个配置关系

c_cpp_properties -> tasks.json -> launch.json

c_cpp_properties是全局配置，tasks.json是项目配置，launch.json是调试配置。

launch.json中的preLaunchTask配置对应tasks.json中label配置。就是launch之前要执行的操作内容。
