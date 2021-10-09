#include "wwdg.h"
#include "stm32f10x.h"
#include "led.h"

u8 WWDG_CNT=0x7f;

 void WWDG_Init(u8 tr,u8 hr,u32 prer)
 {
	 RCC_APB1PeriphClockCmd(RCC_APB1Periph_WWDG, ENABLE);
	
	 WWDG_SetPrescaler(prer);//预分频倍数WWDG_Prescaler_1，1倍，倍数小于8
	 WWDG_SetWindowValue(hr);//设置上窗口值
	 WWDG_Enable(tr); //使能看门狗，设置计数器的初始值
	 WWDG_ClearFlag();//清除提前唤醒中断标志位
	 WWDG_NVIC_Init();//配置看门狗的中断
	WWDG_EnableIT();//开启看门狗中断 	 
 }

//更改计数器的初始值
void WWDG_Set_Counter(u8 st)
{
    WWDG_Enable(st);	 
}
 
 //中断配置函数
void WWDG_NVIC_Init()
{
	NVIC_InitTypeDef NVIC_InitStructure;
	NVIC_InitStructure.NVIC_IRQChannel = WWDG_IRQn;    //WWDG中断
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2;   //抢占2，子优先级3，组2	
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;	 //抢占2，子优先级3，组2	
        NVIC_InitStructure.NVIC_IRQChannelCmd=ENABLE; 
	NVIC_Init(&NVIC_InitStructure);//NVIC初始化
}

void WWDG_IRQHandler(void)
	{
	// Update WWDG counter
	WWDG_SetCounter(0x7F);	  //当禁掉此句后,窗口看门狗将产生复位
	// Clear EWI flag */
	WWDG_ClearFlag();	  //清除提前唤醒中断标志位
	// Toggle GPIO_Led pin 7 */
	LED1=!LED1;
	}
	
