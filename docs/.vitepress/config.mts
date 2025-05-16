import {defineConfig} from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
    title: "Python 3.13.3 教程",
    description: "从基础到高级的 Python 学习文档",
    lang: 'zh-CN',
    lastUpdated: true,
    cleanUrls: true,
    appearance: true,

    head: [
        ['meta', {name: 'theme-color', content: '#3eaf7c'}],
        ['meta', {name: 'apple-mobile-web-app-capable', content: 'yes'}],
        ['meta', {name: 'mobile-web-app-capable', content: 'yes'}],
        ['meta', {name: 'apple-mobile-web-app-status-bar-style', content: 'black'}],
        // 预加载Ubuntu Mono字体以避免FOIT (Flash of Invisible Text)
        ['link', {rel: 'preconnect', href: 'https://fonts.googleapis.com'}],
        ['link', {rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: ''}]
    ],

    // Vite配置
    vite: {
        build: {
            // 增加chunk大小警告限制到1000KB (1MB)
            chunkSizeWarningLimit: 1000
        }
    },

    themeConfig: {
        // https://vitepress.dev/reference/default-theme-config
        // 从主题资源目录使用logo
        // logo: '/logo.svg',
        nav: [
            {text: '首页', link: '/'},
            {text: '教程', link: '/basic/introduction'},
            {text: '进阶主题', link: '/advanced/overview'}
        ],

        sidebar: [
            {
                text: 'Python 基础',
                collapsed: false,
                items: [
                    {text: '介绍', link: '/basic/introduction'},
                    {text: '安装与设置', link: '/basic/installation'},
                    {text: '基本语法', link: '/basic/syntax'},
                    {text: '变量与数据类型', link: '/basic/variables'},
                    {text: '运算符', link: '/basic/operators'},
                    {text: '控制流', link: '/basic/control-flow'},
                    {text: '函数', link: '/basic/functions'},
                    {text: '模块与包', link: '/basic/modules'},
                    {text: '异常处理', link: '/basic/exceptions'}
                ]
            },
            {
                text: 'Python 中级',
                collapsed: false,
                items: [
                    {text: '面向对象编程', link: '/intermediate/oop'},
                    {text: '文件操作', link: '/intermediate/files'},
                    {text: '正则表达式', link: '/intermediate/regex'},
                    {text: '日期与时间', link: '/intermediate/datetime'},
                    {text: '数据结构', link: '/intermediate/data-structures'},
                    {text: '迭代器与生成器', link: '/intermediate/iterators-generators'},
                    {text: '装饰器', link: '/intermediate/decorators'},
                    {text: '上下文管理器', link: '/intermediate/context-managers'}
                ]
            },
            {
                text: 'Python 高级',
                collapsed: false,
                items: [
                    {text: '概述', link: '/advanced/overview'},
                    {text: '多线程与多进程', link: '/advanced/concurrency'},
                    {text: '异步编程', link: '/advanced/async'},
                    {text: '元编程', link: '/advanced/metaprogramming'},
                    {text: '性能优化', link: '/advanced/performance'},
                    {text: '设计模式', link: '/advanced/design-patterns'},
                    {text: '网络编程', link: '/advanced/networking'},
                    {text: 'Python 与数据库', link: '/advanced/databases'},
                    {text: 'Python 3.13 新特性', link: '/advanced/python-3-13-features'}
                ]
            },
            {
                text: '实践项目',
                collapsed: true,
                items: [
                    {text: 'Web 应用开发', link: '/projects/web-app'},
                    {text: '数据分析', link: '/projects/data-analysis'},
                    {text: '机器学习入门', link: '/projects/machine-learning'},
                    {text: '自动化脚本', link: '/projects/automation'}
                ]
            }
        ],

        socialLinks: [
            {icon: 'github', link: 'https://github.com/python/cpython'}
        ],

        footer: {
            message: '基于 MIT 许可发布',
            copyright: 'Copyright © 2023-2024 Python 教程'
        },

        outline: {
            level: 'deep',
            label: '页面导航'
        },

        search: {
            provider: 'local',
            options: {
                locales: {
                    root: {
                        translations: {
                            button: {
                                buttonText: '搜索文档',
                                buttonAriaLabel: '搜索文档'
                            },
                            modal: {
                                noResultsText: '无法找到相关结果',
                                resetButtonTitle: '清除查询条件',
                                footer: {
                                    selectText: '选择',
                                    navigateText: '切换',
                                    closeText: '关闭'
                                }
                            }
                        }
                    }
                }
            }
        },

        docFooter: {
            prev: '上一页',
            next: '下一页'
        }
    }
})
