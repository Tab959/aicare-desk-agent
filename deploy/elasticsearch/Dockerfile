# 在官方8.19长期维护版本上预装中文SmartCN分析器，构建阶段固定插件版本。
FROM docker.elastic.co/elasticsearch/elasticsearch:8.19.18

# --batch显式接受插件权限；镜像构建失败会阻断部署，不在启动时动态安装。
RUN bin/elasticsearch-plugin install --batch analysis-smartcn
