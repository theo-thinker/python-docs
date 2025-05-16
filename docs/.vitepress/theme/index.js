import DefaultTheme from 'vitepress/theme'
import './custom.css'
import { onMounted, watch, h } from 'vue'
import { useRoute } from 'vitepress'
import AnimatedLogo from './components/AnimatedLogo.vue'
import BackToTop from './components/BackToTop.vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app, router }) {
    // 注册全局组件
    app.component('AnimatedLogo', AnimatedLogo)
    app.component('BackToTop', BackToTop)
  },
  // 定制导航栏Logo
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'nav-bar-title-before': () => h(AnimatedLogo),
      'layout-bottom': () => h(BackToTop)
    })
  },
  setup() {
    // 添加页面过渡动画
    const route = useRoute()
    
    onMounted(() => {
      // 添加页面载入动画
      document.body.classList.add('ready')
      
      // 初始化代码块复制功能
      setTimeout(() => {
        const codeBlocks = document.querySelectorAll('pre[class*="language-"]')
        codeBlocks.forEach(block => {
          if (!block.querySelector('.copy-code-button')) {
            const copyButton = document.createElement('button')
            copyButton.className = 'copy-code-button'
            copyButton.textContent = '复制'
            
            copyButton.addEventListener('click', () => {
              const code = block.querySelector('code').innerText
              navigator.clipboard.writeText(code)
              
              copyButton.textContent = '已复制!'
              setTimeout(() => {
                copyButton.textContent = '复制'
              }, 2000)
            })
            
            block.appendChild(copyButton)
          }
        })
      }, 1000)
    })
    
    // 监听路由变化，添加页面过渡效果
    watch(
      () => route.path,
      () => {
        window.scrollTo(0, 0)
        document.body.classList.add('page-transition')
        
        setTimeout(() => {
          document.body.classList.remove('page-transition')
        }, 500)
        
        // 重新初始化代码块复制功能
        setTimeout(() => {
          const codeBlocks = document.querySelectorAll('pre[class*="language-"]')
          codeBlocks.forEach(block => {
            if (!block.querySelector('.copy-code-button')) {
              const copyButton = document.createElement('button')
              copyButton.className = 'copy-code-button'
              copyButton.textContent = '复制'
              
              copyButton.addEventListener('click', () => {
                const code = block.querySelector('code').innerText
                navigator.clipboard.writeText(code)
                
                copyButton.textContent = '已复制!'
                setTimeout(() => {
                  copyButton.textContent = '复制'
                }, 2000)
              })
              
              block.appendChild(copyButton)
            }
          })
        }, 1000)
      }
    )
  }
} 